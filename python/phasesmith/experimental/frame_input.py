"""Optional array frontend for bounded native original-count preparation."""

from phasesmith import _core

from .frame_validity import CombinedValidity, DacApertureMask, FrameValidity


def preparation_plan(shape, frames, vertical_flip, reserved_minimum, validity=None):
    """Snapshot known immutable validity providers into a native preparation plan.

    Expected hashes come from the validity snapshot and are checked on every
    prepared frame. The native implementation computes a fresh digest; this
    adapter never accepts a caller's unchecked current-frame digest.
    """
    hashes, blocks, dacs = [], [], []

    def add(provider):
        if tuple(provider.shape) != tuple(shape) or len(provider.frame_ids) != frames:
            raise ValueError("validity frame count or detector shape differs")
        if type(provider) is CombinedValidity:
            for child in provider.providers:
                add(child)
        elif type(provider) is FrameValidity:
            expected = provider.provenance.get("image_sha256")
            if expected is not None:
                hashes.append(expected)
            blocks.append((provider.block, provider.excluded, provider.fixed_mask))
        elif type(provider) is DacApertureMask:
            dacs.append(
                (
                    provider.geometry.vectors,
                    provider.omega_bounds,
                    provider.pixel_block,
                    tuple(provider.axis_at_omega_zero),
                    provider.half_opening_degrees,
                    provider.fixed_mask,
                )
            )
        else:
            raise TypeError("native frame preparation requires standard validity providers")

    if validity is not None:
        add(validity)
    return _core.NativeFramePreparation(
        shape, frames, vertical_flip, reserved_minimum, hashes, blocks, dacs
    )
