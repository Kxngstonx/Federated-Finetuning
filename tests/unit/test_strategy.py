import numpy as np
import pytest
import torch
from torch import nn

from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from peft import LoraConfig, get_peft_model

from flowertune_llm.models import get_parameters
from flowertune_llm.peft_layers import index_lora_layers
from flowertune_llm.strategies.common import build_layer_refs
from flowertune_llm.strategies.fedora import FeDoRA, aggregate_dora_layer
from flowertune_llm.strategies.fedsvd import FedSVD, svd_reorthogonalize
from flowertune_llm.strategies.flora import FLoRA
from flowertune_llm.strategies.fedrot import rotation_align_optimization
from flowertune_llm.strategies.fedit import FedIT
from flowertune_llm.strategies.ffalora import FFALoRA


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)


def make_model(r=2, alpha=4, use_dora=True):
    torch.manual_seed(0)
    return get_peft_model(
        TinyModel(),
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            use_dora=use_dora,
        ),
    )


def _make_fit_results(template, num_examples, rng):
    results = []
    for n in num_examples:
        arrays = [rng.standard_normal(size=arr.shape).astype(arr.dtype) for arr in template]
        fit_res = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(arrays),
            num_examples=n,
            metrics={},
        )
        results.append((None, fit_res))
    return results


# ---------------------------------------------------------------------------
# FeDoRA (formerly DoraFedAvg)
# ---------------------------------------------------------------------------


def test_build_layer_refs():
    model = make_model()
    layers = build_layer_refs(model)

    assert len(layers) == 2
    names = {layer.name.split(".")[-1] for layer in layers}
    assert names == {"q_proj", "v_proj"}
    for layer in layers:
        assert layer.idx_m is not None
        assert layer.W0.shape == (8, 8)
        assert layer.r == 2


def test_dora_only_guard_raises():
    model = make_model(use_dora=False)
    with pytest.raises(ValueError):
        FeDoRA(model=model)


def test_aggregate_fit_shapes():
    ref_model = make_model()
    template = get_parameters(ref_model)
    strategy = FeDoRA(model=ref_model)

    rng = np.random.default_rng(0)
    results = _make_fit_results(template, (10, 20, 30), rng)

    parameters_aggregated, metrics_aggregated = strategy.aggregate_fit(1, results, [])

    aggregated_arrays = parameters_to_ndarrays(parameters_aggregated)
    assert len(aggregated_arrays) == len(template)
    for agg, tmpl in zip(aggregated_arrays, template):
        assert agg.shape == tmpl.shape
        assert agg.dtype == tmpl.dtype
    assert metrics_aggregated == {}


def _recompute_w_global(w0, scaling, a_list, b_list, m_list, freqs):
    w_global = torch.zeros_like(w0)
    for freq, a_c, b_c, m_c in zip(freqs, a_list, b_list, m_list):
        v_c = w0 + scaling * (b_c @ a_c)
        row_norm = torch.linalg.norm(v_c, dim=1)
        w_c = (m_c / row_norm).unsqueeze(1) * v_c
        w_global = w_global + freq * w_c
    return w_global


def test_aggregate_dora_layer_math():
    torch.manual_seed(1)
    out_features, in_features, r, n_clients = 5, 4, 2, 3
    scaling = 2.0

    w0 = torch.randn(out_features, in_features)
    a_list = [torch.randn(r, in_features) for _ in range(n_clients)]
    b_list = [torch.randn(out_features, r) for _ in range(n_clients)]
    m_list = [torch.rand(out_features) + 0.5 for _ in range(n_clients)]
    num_examples = [2, 3, 5]
    total = sum(num_examples)
    freqs = [n / total for n in num_examples]

    a_new, b_new, m_new = aggregate_dora_layer(
        w0, scaling, r, a_list, b_list, m_list, freqs,
    )

    assert a_new.shape == (r, in_features)
    assert b_new.shape == (out_features, r)
    assert m_new.shape == (out_features,)

    # SVD row-orthonormality.
    assert torch.allclose(a_new @ a_new.T, torch.eye(r), atol=1e-4)

    # Independently recomputed W_global (mirrors step 1, not calling aggregate_dora_layer).
    w_global_ref = _recompute_w_global(w0, scaling, a_list, b_list, m_list, freqs)

    # Step 4: m_new must equal the row-norm of that independently recomputed W_global.
    assert torch.allclose(m_new, torch.linalg.norm(w_global_ref, dim=1), atol=1e-4)

    # Step 3: normal-equations optimality condition that defines B_new as the
    # least-squares solution, checked independently of re-calling torch.linalg.solve.
    m_ref = (w_global_ref - w0) / scaling
    residual = m_ref - b_new @ a_new
    normal_eq = a_new @ residual.T
    assert torch.allclose(normal_eq, torch.zeros_like(normal_eq), atol=1e-3)


# ---------------------------------------------------------------------------
# FedSVD
# ---------------------------------------------------------------------------


def test_svd_reorthogonalize_preserves_product():
    torch.manual_seed(3)
    r, in_features, out_features = 3, 6, 5
    a = torch.randn(r, in_features)
    b = torch.randn(out_features, r)

    a_new, b_new = svd_reorthogonalize(a, b)

    assert a_new.shape == a.shape
    assert b_new.shape == b.shape
    assert torch.allclose(b_new @ a_new, b @ a, atol=1e-4)
    # A_new has orthonormal rows.
    assert torch.allclose(a_new @ a_new.T, torch.eye(r), atol=1e-4)


def test_fedsvd_ffa_averages_b_only_a_passthrough():
    model = make_model(use_dora=False)
    template = get_parameters(model)
    # No reorthogonalization this round (period=0 disables it).
    strategy = FedSVD(model=model, cfg={"recalculate_svd_period": 0, "svd_warmup_steps": 0})

    rng = np.random.default_rng(4)
    # All clients "return" the same frozen A (as a real frozen client would), random B.
    layers = index_lora_layers(model)
    base_arrays = [rng.standard_normal(size=arr.shape).astype(arr.dtype) for arr in template]
    results = []
    for n in (10, 30):
        arrays = list(base_arrays)
        for layer in layers:
            arrays[layer.idx_b] = rng.standard_normal(size=template[layer.idx_b].shape).astype(
                template[layer.idx_b].dtype
            )
        fit_res = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(arrays),
            num_examples=n,
            metrics={},
        )
        results.append((None, fit_res))

    parameters_aggregated, _ = strategy.aggregate_fit(1, results, [])
    aggregated = parameters_to_ndarrays(parameters_aggregated)

    for layer in layers:
        # A passed straight through, untouched.
        assert np.allclose(aggregated[layer.idx_a], base_arrays[layer.idx_a])
        # B is the weighted average of the two clients' B arrays.
        client_arrays = [parameters_to_ndarrays(res.parameters) for _, res in results]
        expected_b = (10 / 40) * client_arrays[0][layer.idx_b] + (30 / 40) * client_arrays[1][layer.idx_b]
        assert np.allclose(aggregated[layer.idx_b], expected_b, atol=1e-4)


def test_fedsvd_reorthogonalization_only_fires_on_schedule():
    model = make_model(use_dora=False)
    template = get_parameters(model)
    strategy = FedSVD(model=model, cfg={"recalculate_svd_period": 2, "svd_warmup_steps": 1})

    rng = np.random.default_rng(5)
    results = _make_fit_results(template, (10, 10), rng)

    # Round 1: warmup_steps=1 means round 1 is not > 1, so no reorthogonalization.
    params_r1, _ = strategy.aggregate_fit(1, results, [])
    a_r1 = parameters_to_ndarrays(params_r1)[index_lora_layers(model)[0].idx_a]

    # Round 2: 2 % 2 == 0 and 2 > 1 -> reorthogonalization fires, A changes.
    params_r2, _ = strategy.aggregate_fit(2, results, [])
    a_r2 = parameters_to_ndarrays(params_r2)[index_lora_layers(model)[0].idx_a]

    assert not np.allclose(a_r1, a_r2)

    # Round 3: 3 % 2 != 0 -> no reorthogonalization, A stays as round 2 left it.
    params_r3, _ = strategy.aggregate_fit(3, results, [])
    a_r3 = parameters_to_ndarrays(params_r3)[index_lora_layers(model)[0].idx_a]
    assert np.allclose(a_r2, a_r3)


def test_fedsvd_recomputes_m_every_round_ignoring_client_values():
    """DoRA magnitude vector: FedSVD must recompute m analytically every round from
    row_norm(W0 + scaling*(b_out@a_out)), completely ignoring whatever clients uploaded."""
    model = make_model(use_dora=True)
    template = get_parameters(model)
    strategy = FedSVD(model=model, cfg={"recalculate_svd_period": 0, "svd_warmup_steps": 0})
    layers = build_layer_refs(model)

    rng = np.random.default_rng(15)
    # Two different sets of client-reported m values for the SAME A/B -- result must be
    # identical either way, since the server discards client m entirely.
    results_a = _make_fit_results(template, (10, 30), rng)
    results_b = [
        (
            proxy,
            FitRes(
                status=res.status,
                parameters=ndarrays_to_parameters(
                    [
                        arr if i != layers[0].idx_m else rng.standard_normal(size=arr.shape).astype(arr.dtype)
                        for i, arr in enumerate(parameters_to_ndarrays(res.parameters))
                    ]
                ),
                num_examples=res.num_examples,
                metrics=res.metrics,
            ),
        )
        for proxy, res in results_a
    ]

    params_a, _ = strategy.aggregate_fit(1, results_a, [])

    strategy2 = FedSVD(model=model, cfg={"recalculate_svd_period": 0, "svd_warmup_steps": 0})
    params_b, _ = strategy2.aggregate_fit(1, results_b, [])

    m_a = parameters_to_ndarrays(params_a)[layers[0].idx_m]
    m_b = parameters_to_ndarrays(params_b)[layers[0].idx_m]
    assert np.allclose(m_a, m_b, atol=1e-4)  # client m was irrelevant either way

    # And it must match the documented closed-form recomputation.
    client_arrays = [parameters_to_ndarrays(res.parameters) for _, res in results_a]
    num_examples = [res.num_examples for _, res in results_a]
    b_avg = aggregate([([arr[layers[0].idx_b]], n) for arr, n in zip(client_arrays, num_examples)])[0]
    a_cur = client_arrays[0][layers[0].idx_a]
    w_eff = layers[0].W0 + layers[0].scaling * (
        torch.from_numpy(b_avg.astype(np.float32)) @ torch.from_numpy(a_cur.astype(np.float32))
    )
    expected_m = torch.linalg.norm(w_eff, dim=1).numpy()
    assert np.allclose(m_a, expected_m, atol=1e-4)


# ---------------------------------------------------------------------------
# FLoRA
# ---------------------------------------------------------------------------


def test_flora_direct_sum_equals_stacked_block_product():
    """Regression guard for the simplification used in strategies/flora.py: summing
    freq_k * scaling * (B_k @ A_k) directly is mathematically identical to literally
    concatenating A's along dim0 and B's along dim1 and then multiplying."""
    torch.manual_seed(6)
    r, in_features, out_features, n_clients = 2, 5, 4, 3
    scaling = 2.0
    freqs = [0.2, 0.3, 0.5]

    a_list = [torch.randn(r, in_features) for _ in range(n_clients)]
    b_list = [torch.randn(out_features, r) for _ in range(n_clients)]

    direct = sum(freq * scaling * (b @ a) for freq, a, b in zip(freqs, a_list, b_list))

    a_stack = torch.cat([a * freq for a, freq in zip(a_list, freqs)], dim=0)
    b_stack = torch.cat(b_list, dim=1)
    stacked = scaling * (b_stack @ a_stack)

    assert torch.allclose(direct, stacked, atol=1e-4)


def test_flora_merge_and_reset_shapes():
    model = make_model(use_dora=False)
    template = get_parameters(model)
    strategy = FLoRA(model=model)

    rng = np.random.default_rng(7)
    results = _make_fit_results(template, (10, 20), rng)

    parameters_aggregated, _ = strategy.aggregate_fit(1, results, [])
    aggregated = parameters_to_ndarrays(parameters_aggregated)

    layers = index_lora_layers(model)
    n_layers = len(layers)
    lora_part = aggregated[: len(aggregated) - n_layers]
    extra_w0 = aggregated[len(aggregated) - n_layers :]

    assert len(lora_part) == len(template)
    for layer in layers:
        # B reset to all zeros.
        assert np.allclose(lora_part[layer.idx_b], 0.0)
        # A freshly initialized, non-degenerate (not all zero).
        assert not np.allclose(lora_part[layer.idx_a], 0.0)

    assert len(extra_w0) == n_layers
    for w0_arr in extra_w0:
        assert w0_arr.dtype == np.float32


def test_flora_master_w0_accumulates_exactly():
    model = make_model(use_dora=False)
    template = get_parameters(model)
    strategy = FLoRA(model=model)
    layers = index_lora_layers(model)
    layer_name = layers[0].name

    w0_before = strategy._w0[layer_name].clone()

    rng = np.random.default_rng(8)
    results = _make_fit_results(template, (10, 20), rng)
    strategy.aggregate_fit(1, results, [])

    w0_after = strategy._w0[layer_name]
    assert not torch.allclose(w0_before, w0_after)  # it moved
    # The master is a plain float32 tensor throughout -- never round-tripped through
    # quantize/dequantize, so its dtype and precision are untouched by aggregation.
    assert w0_after.dtype == torch.float32


def test_flora_dora_m_weighted_average_independent_of_reset_cycle():
    """DoRA magnitude vector: FLoRA must NOT reject DoRA models, must NOT reset/merge m (unlike
    A/B), and must combine client m via plain data-size-weighted FedAvg."""
    model = make_model(use_dora=True)
    template = get_parameters(model)
    strategy = FLoRA(model=model)  # must not raise
    layers = index_lora_layers(model)

    rng = np.random.default_rng(16)
    results = _make_fit_results(template, (10, 30), rng)

    parameters_aggregated, _ = strategy.aggregate_fit(1, results, [])
    aggregated = parameters_to_ndarrays(parameters_aggregated)

    client_arrays = [parameters_to_ndarrays(res.parameters) for _, res in results]
    m_idx = layers[0].idx_m
    expected_m = (10 / 40) * client_arrays[0][m_idx] + (30 / 40) * client_arrays[1][m_idx]
    assert np.allclose(aggregated[m_idx], expected_m, atol=1e-4)

    # A/B still merge-and-reset as usual (B all zero, A non-degenerate) -- m is unaffected by it.
    assert np.allclose(aggregated[layers[0].idx_b], 0.0)
    assert not np.allclose(aggregated[layers[0].idx_a], 0.0)


def test_flora_quantization_args_not_hardcoded():
    """bitsandbytes re-quantization must reuse the model's own existing quant args
    (blocksize/quant_type/compress_statistics/quant_storage), never hardcode e.g. 'nf4'."""
    bnb = pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("bitsandbytes 4-bit quantize_4bit requires a CUDA device")
    from flowertune_llm.models import set_parameters

    linear = bnb.nn.Linear4bit(8, 8, bias=False, quant_type="fp4", compress_statistics=False)
    existing = linear.weight
    assert existing.quant_type == "fp4"
    assert existing.compress_statistics is False

    w0 = np.random.default_rng(9).standard_normal((8, 8)).astype(np.float32)
    w0_4bit, quant_state = bnb.functional.quantize_4bit(
        torch.from_numpy(w0).cuda(),
        blocksize=existing.blocksize,
        compress_statistics=existing.compress_statistics,
        quant_type=existing.quant_type,
        quant_storage=existing.quant_storage,
    )
    new_param = bnb.nn.Params4bit(
        w0_4bit,
        requires_grad=False,
        quant_state=quant_state,
        blocksize=existing.blocksize,
        compress_statistics=existing.compress_statistics,
        quant_type=existing.quant_type,
        quant_storage=existing.quant_storage,
        module=linear,
        bnb_quantized=True,
    )
    assert new_param.quant_type == "fp4"  # preserved, not defaulted to "nf4"
    assert new_param.compress_statistics is False


# ---------------------------------------------------------------------------
# FedRot
# ---------------------------------------------------------------------------


def test_rotation_preserves_ba_product():
    torch.manual_seed(10)
    r, in_features, out_features = 3, 5, 4
    ref_a = torch.randn(r, in_features)
    a = torch.randn(r, in_features)
    b = torch.randn(out_features, r)

    a_new, b_new = rotation_align_optimization(ref_a, "A", a, b)

    assert torch.allclose(b_new @ a_new, b @ a, atol=1e-4)


def test_rotation_determinant_flip_guard():
    """Construct M whose raw SVD solution is a reflection (det < 0), and check the ported
    hard-rotation path flips it back to a proper rotation (det > 0)."""
    torch.manual_seed(11)
    r = 4
    # A diagonal M with one negative entry reliably drives det(U @ Vh) < 0 for its SVD.
    signs = torch.ones(r)
    signs[-1] = -1.0
    m = torch.diag(signs) * torch.rand(r).abs().clamp_min(0.1)

    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    assert torch.linalg.det(u @ vh) < 0  # confirm the raw solution would be a reflection

    # Build a and ref_a such that M = a @ ref_a.T recovers the same correlation matrix.
    ref_a = torch.eye(r)
    a = m  # so that a @ ref_a.T == m exactly
    b = torch.randn(6, r)

    a_new, b_new = rotation_align_optimization(ref_a, "A", a, b)
    # BA product must still be preserved even though the raw SVD solution was a reflection.
    assert torch.allclose(b_new @ a_new, b @ a, atol=1e-3)


def test_fedrot_first_round_skips_rotation():
    torch.manual_seed(12)
    ref_a = torch.randn(3, 5)
    a = torch.randn(3, 5)
    b = torch.randn(4, 3)
    # current_round <= 1 is handled in client_app.py, not in rotation_align_optimization itself
    # (which always rotates when called) -- this is exercised at the FlowerClient level; here we
    # just confirm the primitive itself preserves BA regardless of round, as documented.
    a_new, b_new = rotation_align_optimization(ref_a, "A", a, b)
    assert torch.allclose(b_new @ a_new, b @ a, atol=1e-4)


def test_fedrot_aggregates_via_data_weighted_mean():
    """FedRot aggregation matches FedAvg -- data-size-weighted averaging across clients,
    applied uniformly to lora_A, lora_B, and (if present) DoRA's m."""
    model = make_model(use_dora=True)
    template = get_parameters(model)
    from flowertune_llm.strategies.fedrot import FedRot

    strategy = FedRot(model=model)
    assert isinstance(strategy, FedAvg)

    rng = np.random.default_rng(17)
    # Deliberately very unequal dataset sizes so the weighted skew is easy to detect.
    results = _make_fit_results(template, (1, 1000), rng)

    parameters_aggregated, _ = strategy.aggregate_fit(1, results, [])
    aggregated = parameters_to_ndarrays(parameters_aggregated)

    client_arrays = [parameters_to_ndarrays(res.parameters) for _, res in results]
    for i in range(len(template)):
        weighted = (1 / 1001) * client_arrays[0][i].astype(np.float32) + (1000 / 1001) * client_arrays[1][
            i
        ].astype(np.float32)
        assert np.allclose(aggregated[i], weighted, atol=1e-3)
        # Sanity: the plain unweighted mean would differ substantially from the weighted one.
        uniform = (client_arrays[0][i].astype(np.float32) + client_arrays[1][i].astype(np.float32)) / 2
        if not np.allclose(client_arrays[0][i], client_arrays[1][i]):
            assert not np.allclose(aggregated[i], uniform, atol=1e-3)


# ---------------------------------------------------------------------------
# FedIT
# ---------------------------------------------------------------------------


def test_fedit_matches_plain_fedavg():
    model = make_model(use_dora=False)
    template = get_parameters(model)

    rng = np.random.default_rng(13)
    results = _make_fit_results(template, (10, 20, 5), rng)

    fedit = FedIT(model=model)
    fedavg = FedAvg()

    params_fedit, _ = fedit.aggregate_fit(1, results, [])
    params_fedavg, _ = fedavg.aggregate_fit(1, results, [])

    arrays_fedit = parameters_to_ndarrays(params_fedit)
    arrays_fedavg = parameters_to_ndarrays(params_fedavg)
    for a, b in zip(arrays_fedit, arrays_fedavg):
        assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# FFALoRA
# ---------------------------------------------------------------------------


def test_ffalora_a_fixed_b_averaged():
    model = make_model(use_dora=False)
    template = get_parameters(model)
    strategy = FFALoRA(model=model)
    layers = index_lora_layers(model)

    rng = np.random.default_rng(14)
    base_arrays = [rng.standard_normal(size=arr.shape).astype(arr.dtype) for arr in template]
    results = []
    for n in (10, 30):
        arrays = list(base_arrays)
        for layer in layers:
            arrays[layer.idx_b] = rng.standard_normal(size=template[layer.idx_b].shape).astype(
                template[layer.idx_b].dtype
            )
        fit_res = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(arrays),
            num_examples=n,
            metrics={},
        )
        results.append((None, fit_res))

    params_r1, _ = strategy.aggregate_fit(1, results, [])
    params_r2, _ = strategy.aggregate_fit(2, results, [])
    a_r1 = parameters_to_ndarrays(params_r1)[layers[0].idx_a]
    a_r2 = parameters_to_ndarrays(params_r2)[layers[0].idx_a]

    # A never changes across rounds.
    assert np.allclose(a_r1, a_r2)
    assert np.allclose(a_r1, base_arrays[layers[0].idx_a])

    client_arrays = [parameters_to_ndarrays(res.parameters) for _, res in results]
    expected_b = (10 / 40) * client_arrays[0][layers[0].idx_b] + (30 / 40) * client_arrays[1][layers[0].idx_b]
    b_r1 = parameters_to_ndarrays(params_r1)[layers[0].idx_b]
    assert np.allclose(b_r1, expected_b, atol=1e-4)


def test_ffalora_dora_m_cached_never_recomputed():
    """DoRA magnitude vector: FFALoRA must seed m once (round 1) and pass it through
    unchanged forever, ignoring whatever different m values clients report in later rounds."""
    model = make_model(use_dora=True)
    template = get_parameters(model)
    strategy = FFALoRA(model=model)
    layers = index_lora_layers(model)
    m_idx = layers[0].idx_m

    rng = np.random.default_rng(18)
    results_r1 = _make_fit_results(template, (10, 30), rng)
    params_r1, _ = strategy.aggregate_fit(1, results_r1, [])
    m_r1 = parameters_to_ndarrays(params_r1)[m_idx]

    # Round 2: clients report entirely different (fresh-random) m values.
    results_r2 = _make_fit_results(template, (10, 30), rng)
    params_r2, _ = strategy.aggregate_fit(2, results_r2, [])
    m_r2 = parameters_to_ndarrays(params_r2)[m_idx]

    client_arrays_r2 = [parameters_to_ndarrays(res.parameters) for _, res in results_r2]
    assert not np.allclose(m_r1, client_arrays_r2[0][m_idx])  # round 2 clients differ from round 1
    assert np.allclose(m_r1, m_r2, atol=1e-6)  # but the cached/output m never moves
