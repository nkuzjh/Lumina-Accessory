"""Full parameter and optimizer coverage audit for the aligned Accessory run."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from torch import nn

from models_accessory.lora import LinearLora


def _role(name: str) -> str:
    name = name.replace("dit.module.", "dit.", 1)
    if name.startswith("dit.layers."):
        return "unified_generation_transformer"
    if name.startswith("dit.noise_refiner."):
        return "target_latent_refiner"
    if name.startswith("dit.context_refiner.") or name.startswith("dit.cap_embedder."):
        return "text_feature_processing"
    if name.startswith("dit.cond_refiner.") or name.startswith("dit.cond_embedder."):
        return "radar_condition_processing"
    if name.startswith("dit."):
        return "generation_input_output"
    if name.startswith("text_encoder."):
        return "frozen_gemma"
    if name.startswith("vae."):
        return "frozen_flux_vae"
    return "unknown"


def audit_trainable(bundle_or_model, optimizer=None, output_dir=None, *, expect_official=True) -> dict:
    """Enumerate every tensor and prove optimizer covers exactly the FP32 LoRA set."""
    if isinstance(bundle_or_model, nn.Module):
        components = {"dit": bundle_or_model}
    else:
        components = {n: getattr(bundle_or_model, n) for n in ("dit", "text_encoder", "vae")}
    parameter_ids = {}
    rows = []
    opt_ids = []
    opt_groups = {}
    if optimizer is not None:
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter in group["params"]:
                opt_ids.append(id(parameter))
                opt_groups[id(parameter)] = {"index": group_index, "lr": group.get("lr"), "weight_decay": group.get("weight_decay")}
        duplicates = [pid for pid, count in Counter(opt_ids).items() if count != 1]
        if duplicates:
            raise AssertionError(f"Optimizer duplicates {len(duplicates)} parameters")
    for component_name, component in components.items():
        for name, parameter in component.named_parameters():
            full_name = f"{component_name}.{name}"
            pid = id(parameter)
            if pid in parameter_ids:
                raise AssertionError(f"Parameter alias: {full_name} and {parameter_ids[pid]}")
            parameter_ids[pid] = full_name
            lora = ".lora_A." in full_name or ".lora_B." in full_name
            if parameter.requires_grad != lora:
                raise AssertionError(f"Freeze/LoRA mismatch: {full_name}")
            if lora and str(parameter.dtype) != "torch.float32":
                raise AssertionError(f"LoRA must be FP32: {full_name}")
            in_optimizer = pid in opt_groups
            if optimizer is not None and in_optimizer != lora:
                raise AssertionError(f"Optimizer coverage mismatch: {full_name}")
            rows.append({
                "name": full_name, "shape": list(parameter.shape), "numel": parameter.numel(),
                "dtype": str(parameter.dtype), "device": str(parameter.device), "trainable": parameter.requires_grad,
                "category": "lora" if lora else "frozen", "function": _role(full_name),
                "optimizer_included": in_optimizer,
                "optimizer_group": opt_groups.get(pid),
            })
    if optimizer is not None and set(opt_ids) != {id(p) for c in components.values() for p in c.parameters() if p.requires_grad}:
        raise AssertionError("Optimizer includes a parameter outside the audited bundle")
    targets = []
    for name, module in components["dit"].named_modules():
        if isinstance(module, LinearLora):
            targets.append({"name": name, "rank": module.rank, "in_features": module.in_features,
                            "out_features": module.out_features, "base_has_bias": module.bias is not None,
                            "b_has_bias": module.lora_B.bias is not None})
    groups = Counter()
    for row in rows:
        if row["trainable"]:
            groups[row["function"]] += row["numel"]
    trainable_count = sum(r["numel"] for r in rows if r["trainable"])
    dit_count = sum(r["numel"] for r in rows if r["name"].startswith("dit."))
    bundle_count = sum(r["numel"] for r in rows)
    summary = {
        "linear_targets": len(targets),
        "trainable_parameters": trainable_count,
        "base_dit_parameters": dit_count - trainable_count,
        "dit_parameters": dit_count,
        "total_parameters": bundle_count,
        "trainable_fraction_dit": trainable_count / dit_count,
        "trainable_fraction_bundle": trainable_count / bundle_count,
        "bundle_complete": not isinstance(bundle_or_model, nn.Module),
        "trainable_by_function": dict(groups),
        "optimizer_verified": optimizer is not None,
        "meta_only": all(parameter.device.type == "meta" for component in components.values() for parameter in component.parameters()),
    }
    if expect_official and (summary["linear_targets"], summary["trainable_parameters"]) != (197, 227963968):
        raise AssertionError(f"Official native LoRA audit differs from reviewed baseline: {summary}")
    report = {"summary": summary, "lora_targets": targets, "parameters": rows,
              "module_mapping": {"UniLIP_LLM": "frozen Gemma encoder + single context_refiner/cap_embedder LoRA",
                                 "UniLIP_generation_head": "Accessory unified layers/noise_refiner/final_layer LoRA",
                                 "UniLIP_radar_connector": "Accessory cond_embedder/cond_refiner LoRA",
                                 "UniLIP_vision": "frozen FLUX VAE"}}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "trainable_parameter_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [f"Meta-only architecture audit: {summary['meta_only']}",
                 f"Bundle complete (DiT, Gemma, VAE): {summary['bundle_complete']}",
                 f"Targets: {len(targets)}; trainable: {trainable_count:,}; frozen base DiT: {summary['base_dit_parameters']:,}; DiT total: {dit_count:,}; bundle total: {bundle_count:,}",
                 f"Trainable / DiT total: {summary['trainable_fraction_dit']:.6%}; trainable / bundle total: {summary['trainable_fraction_bundle']:.6%}",
                 f"Optimizer coverage verified: {summary['optimizer_verified']}", "", "Trainable by function:"]
        lines += [f"- {name}: {count:,}" for name, count in sorted(groups.items())]
        lines += ["", "Native Linear LoRA targets:"]
        lines += [f"- {t['name']}: rank={t['rank']} in={t['in_features']} out={t['out_features']} base_bias={t['base_has_bias']} B_bias={t['b_has_bias']}" for t in targets]
        lines += ["", f"All parameter tensors ({len(rows)} rows):", "",
                  "| Parameter | Shape | Elements | State | Function | LR | Weight decay | Optimizer |",
                  "| --- | --- | ---: | --- | --- | ---: | ---: | --- |"]
        for row in rows:
            group = row["optimizer_group"]
            lr = f"{group['lr']:.8g}" if group is not None and group["lr"] is not None else "—"
            weight_decay = f"{group['weight_decay']:.8g}" if group is not None and group["weight_decay"] is not None else "—"
            shape = " × ".join(str(dim) for dim in row["shape"]) or "scalar"
            lines.append(f"| `{row['name']}` | {shape} | {row['numel']:,} | {row['category']} | {row['function']} | {lr} | {weight_decay} | {'yes' if row['optimizer_included'] else 'no'} |")
        (output / "trainable_parameter_audit.md").write_text("\n".join(lines) + "\n")
    return report
