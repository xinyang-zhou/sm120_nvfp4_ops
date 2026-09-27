"""Plan inspection for the reviewed FlashInfer public sparse-MLA API."""


def inspect_public_plan(q, swa_cache, swa_ids, output, swa_lengths, sink,
                        comp_cache, comp_ids, comp_lengths):
    # q/output use [B,1,H,D], as in the public single-token decode contract.
    # Keep this diagnostic paired with the reviewed-revision environment gate.
    from flashinfer.mla._sparse_mla_sm120 import _prepared

    tensors = (q[:, 0], swa_cache, swa_ids, output[:, 0], swa_lengths, sink,
               comp_cache, comp_ids, comp_lengths, None, None, None)
    prepared = _prepared._functional_plan(tensors, 1, True, False)
    info = dict(prepared.plan.inspect())
    text_fields = {"numeric_route", "implementation", "merge"}
    return {str(k): str(v) if str(k) in text_fields else int(v) for k, v in info.items()}
