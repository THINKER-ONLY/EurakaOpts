import importlib.util
import math
import traceback

import torch


CANDIDATE = "/data/tilelang-metax/contest_fused_moe_gemm/v033_serial_fc1_accumulator.py"
REPEATS = 35
BLOCK_TOKEN = 128


def load_candidate():
    spec = importlib.util.spec_from_file_location("candidate", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_metadata(group_sum, n_experts, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(81394)
    assignments = torch.randint(
        0,
        n_experts,
        (group_sum,),
        generator=generator,
        dtype=torch.int64,
    )
    counts = torch.bincount(assignments, minlength=n_experts).tolist()

    group_offsets = []
    group_padded_offsets = []
    raw_offset = 0
    padded_offset = 0
    for count in counts:
        group_offsets.append(raw_offset)
        group_padded_offsets.append(padded_offset)
        raw_offset += count
        padded_offset += math.ceil((count + 1) / BLOCK_TOKEN) * BLOCK_TOKEN

    group_idx = []
    expert_id = 0
    for bx in range(padded_offset // BLOCK_TOKEN):
        padded_start = bx * BLOCK_TOKEN
        while (
            expert_id + 1 < n_experts
            and padded_start >= group_padded_offsets[expert_id + 1]
        ):
            expert_id += 1
        group_idx.append(expert_id)

    return (
        counts,
        group_padded_offsets,
        padded_offset,
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(group_offsets, dtype=torch.int32, device=device),
        torch.tensor(group_padded_offsets, dtype=torch.int32, device=device),
        torch.tensor(group_idx, dtype=torch.int32, device=device),
    )


def checksum(tensor):
    values = tensor.float()
    return float(values.sum().item()), float(values.square().sum().item())


def main():
    n_experts = 16
    d_hidden = 2048
    d_expert = 8192
    group_sum = 2272
    device = "cuda"

    torch.manual_seed(81394)
    (
        counts,
        group_padded_offsets,
        padded_sum,
        group_sizes,
        group_offsets,
        padded_offsets,
        group_idx,
    ) = build_metadata(group_sum, n_experts, device)

    x = torch.randn((padded_sum, d_hidden), dtype=torch.float16, device=device)
    gate_w = torch.randn(
        (n_experts, d_expert, d_hidden), dtype=torch.float16, device=device
    ) / math.sqrt(d_expert)
    up_w = torch.randn_like(gate_w) / math.sqrt(d_expert)
    down_w = torch.randn(
        (n_experts, d_hidden, d_expert), dtype=torch.float16, device=device
    ) / math.sqrt(d_hidden)
    route_w = torch.rand((group_sum,), dtype=torch.float16, device=device)
    out = torch.empty((padded_sum, d_hidden), dtype=torch.float16, device=device)

    readonly = [
        x,
        gate_w,
        up_w,
        down_w,
        route_w,
        group_sizes,
        group_offsets,
        padded_offsets,
        group_idx,
    ]
    readonly_checksums = [checksum(tensor) for tensor in readonly[:5]]
    metadata_copies = [tensor.clone() for tensor in readonly[5:]]
    module = load_candidate()
    baseline = None

    print(
        "REPEAT_STABILITY_BEGIN",
        REPEATS,
        n_experts,
        d_hidden,
        d_expert,
        group_sum,
        padded_sum,
        flush=True,
    )
    for launch in range(REPEATS):
        out.fill_(7)
        module.run_kernel(*readonly, out)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all().item())

        padding_max = 0.0
        for expert_id, count in enumerate(counts):
            padded_start = group_padded_offsets[expert_id]
            padded_end = (
                group_padded_offsets[expert_id + 1]
                if expert_id + 1 < n_experts
                else padded_sum
            )
            if padded_start + count < padded_end:
                padding_max = max(
                    padding_max,
                    float(out[padded_start + count : padded_end].abs().max().item()),
                )
        assert padding_max == 0.0

        current_checksum = checksum(out)
        if baseline is None:
            baseline = out.clone()
            baseline_checksum = current_checksum
            drift = 0.0
        else:
            drift = float((out.float() - baseline.float()).abs().max().item())
            assert drift == 0.0
            assert current_checksum == baseline_checksum

        print(
            "LAUNCH_PASS",
            launch + 1,
            current_checksum[0],
            current_checksum[1],
            padding_max,
            drift,
            flush=True,
        )

    assert readonly_checksums == [checksum(tensor) for tensor in readonly[:5]]
    for tensor, copy in zip(readonly[5:], metadata_copies):
        assert torch.equal(tensor, copy)
    print("REPEAT_STABILITY_PASS", REPEATS, flush=True)
    print("READONLY_INPUTS_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
