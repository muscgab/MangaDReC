"""Device-resident Torch fallback for smart DB geometry.

The kernels are intentionally expressed with a fixed number of launches and no
device-to-host decisions.  CUDA uses this implementation until the same logic
is folded into the native CCL extension; MPS uses the lower-overhead Metal
implementation in :mod:`smart_mps`.
"""

from __future__ import annotations

import torch


def adaptive_expand_torch(
    probability: torch.Tensor,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    *,
    threshold: float,
    maximum_support_area_ratio: float,
    minimum_pixels: int,
    maximum_gap: float,
    minimum_extent: float,
    support_margin: float,
    cross_cap_ratio: float,
    end_cap_ratio: float,
    minimum_cap: float,
    maximum_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch implementation of conservative weak-support expansion."""
    if probability.device.type not in {"cuda", "mps"}:
        raise ValueError("adaptive expansion requires a CUDA or MPS tensor")
    source = probability.contiguous().float()[:, 0]
    source_boxes = boxes.contiguous().float()
    source_valid = valid.contiguous()
    batch, height, width = source.shape
    capacity = source_boxes.shape[1]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=torch.float32),
        torch.arange(width, device=source.device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    active = source >= threshold
    output_slots: list[torch.Tensor] = []
    expanded_slots: list[torch.Tensor] = []
    infinity = torch.tensor(float("inf"), device=source.device)

    for slot in range(capacity):
        box = source_boxes[:, slot]
        centre = box.mean(dim=1)
        horizontal = (box[:, 1] - box[:, 0]) + (box[:, 2] - box[:, 3])
        vertical = (box[:, 3] - box[:, 0]) + (box[:, 2] - box[:, 1])
        horizontal = horizontal / horizontal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vertical = vertical / vertical.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        relative_corners = box - centre[:, None]
        box_u = (relative_corners * horizontal[:, None]).sum(dim=-1)
        box_v = (relative_corners * vertical[:, None]).sum(dim=-1)
        minimum_u, maximum_u = box_u.amin(dim=1), box_u.amax(dim=1)
        minimum_v, maximum_v = box_v.amin(dim=1), box_v.amax(dim=1)
        box_width = maximum_u - minimum_u
        box_height = maximum_v - minimum_v
        short_side = torch.minimum(box_width, box_height)
        vertical_text = box_height > box_width * 1.05
        cross_cap = (cross_cap_ratio * short_side).clamp(minimum_cap, maximum_cap)
        end_cap = (end_cap_ratio * short_side).clamp(minimum_cap, maximum_cap)
        caps = torch.stack(
            [
                torch.where(vertical_text, cross_cap, end_cap),
                torch.where(vertical_text, cross_cap, end_cap),
                torch.where(vertical_text, end_cap, cross_cap),
                torch.where(vertical_text, end_cap, cross_cap),
            ],
            dim=1,
        )
        relative_x = xx - centre[:, 0, None, None]
        relative_y = yy - centre[:, 1, None, None]
        local_u = (
            relative_x * horizontal[:, 0, None, None]
            + relative_y * horizontal[:, 1, None, None]
        )
        local_v = (
            relative_x * vertical[:, 0, None, None]
            + relative_y * vertical[:, 1, None, None]
        )
        distances = (
            minimum_u[:, None, None] - local_u,
            local_u - maximum_u[:, None, None],
            minimum_v[:, None, None] - local_v,
            local_v - maximum_v[:, None, None],
        )
        orthogonal = (local_v, local_v, local_u, local_u)
        orthogonal_minimum = (minimum_v, minimum_v, minimum_u, minimum_u)
        orthogonal_maximum = (maximum_v, maximum_v, maximum_u, maximum_u)
        changes: list[torch.Tensor] = []
        triggers: list[torch.Tensor] = []
        maximum_support_pixels = maximum_support_area_ratio * short_side.square()
        for side in range(4):
            span = orthogonal_maximum[side] - orthogonal_minimum[side]
            selection = (
                active
                & (distances[side] > 0.0)
                & (distances[side] <= caps[:, side, None, None])
                & (
                    orthogonal[side]
                    >= orthogonal_minimum[side][:, None, None]
                    + 0.05 * span[:, None, None]
                )
                & (
                    orthogonal[side]
                    <= orthogonal_maximum[side][:, None, None]
                    - 0.05 * span[:, None, None]
                )
            )
            count = selection.sum(dim=(-2, -1))
            nearest = torch.where(selection, distances[side], infinity).amin(
                dim=(-2, -1)
            )
            farthest = torch.where(selection, distances[side], -infinity).amax(
                dim=(-2, -1)
            )
            trigger = (
                source_valid[:, slot]
                & (count >= minimum_pixels)
                & (count <= maximum_support_pixels)
                & (nearest <= maximum_gap)
                & (farthest >= minimum_extent)
            )
            changes.append(
                torch.where(
                    trigger,
                    torch.minimum(farthest + support_margin, caps[:, side]),
                    torch.zeros_like(farthest),
                )
            )
            triggers.append(trigger)
        change = torch.stack(changes, dim=1)
        expanded = torch.stack(triggers, dim=1)
        new_minimum_u = minimum_u - change[:, 0]
        new_maximum_u = maximum_u + change[:, 1]
        new_minimum_v = minimum_v - change[:, 2]
        new_maximum_v = maximum_v + change[:, 3]
        local_u_corners = torch.stack(
            [new_minimum_u, new_maximum_u, new_maximum_u, new_minimum_u], dim=1
        )
        local_v_corners = torch.stack(
            [new_minimum_v, new_minimum_v, new_maximum_v, new_maximum_v], dim=1
        )
        refined = (
            centre[:, None]
            + local_u_corners[:, :, None] * horizontal[:, None]
            + local_v_corners[:, :, None] * vertical[:, None]
        )
        refined_x = refined[:, :, 0].clamp(0.0, float(width))
        refined_y = refined[:, :, 1].clamp(0.0, float(height))
        refined = torch.stack([refined_x, refined_y], dim=-1)
        any_change = expanded.any(dim=1)[:, None, None]
        output_slots.append(torch.where(any_change, refined, box))
        expanded_slots.append(expanded)
    return torch.stack(output_slots, dim=1), torch.stack(expanded_slots, dim=1)


def reading_order_torch(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    *,
    robust: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer global orientation and a deterministic pairwise reading order."""
    source_boxes = boxes.contiguous().float()
    source_valid = valid.contiguous()
    batch, capacity = source_valid.shape
    centre = source_boxes.mean(dim=2)
    horizontal = (source_boxes[:, :, 1] - source_boxes[:, :, 0]) + (
        source_boxes[:, :, 2] - source_boxes[:, :, 3]
    )
    vertical_axis = (source_boxes[:, :, 3] - source_boxes[:, :, 0]) + (
        source_boxes[:, :, 2] - source_boxes[:, :, 1]
    )
    widths = (source_boxes[:, :, 1] - source_boxes[:, :, 0]).norm(
        dim=-1
    ).clamp_min(1e-6)
    heights = (source_boxes[:, :, 3] - source_boxes[:, :, 0]).norm(
        dim=-1
    ).clamp_min(1e-6)
    raw_areas = (widths * heights).clamp_min(1.0)
    areas = raw_areas * source_valid
    if robust:
        ordered_areas = torch.where(
            source_valid,
            raw_areas,
            torch.full_like(raw_areas, float("inf")),
        ).sort(dim=1).values
        valid_count = source_valid.sum(dim=1)
        median_index = ((valid_count - 1).clamp_min(0) // 2).to(torch.int64)
        median_area = ordered_areas.gather(1, median_index[:, None])[:, 0]
        aspect = torch.maximum(widths, heights) / torch.minimum(widths, heights)
        reliable = (
            source_valid
            & (aspect >= 1.25)
            & (raw_areas >= 0.15 * median_area[:, None])
        )
        reliable = torch.where(
            reliable.any(dim=1)[:, None], reliable, source_valid
        )
        orientation_areas = raw_areas * reliable
    else:
        orientation_areas = areas
    orientation_vote = (
        orientation_areas * torch.log((heights + 1.0) / (widths + 1.0))
    ).sum(dim=1)
    is_vertical = orientation_vote >= 0.0
    horizontal = horizontal / horizontal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vertical_axis = vertical_axis / vertical_axis.norm(dim=-1, keepdim=True).clamp_min(
        1e-6
    )
    chosen_axis = torch.where(is_vertical[:, None, None], vertical_axis, horizontal)
    sign_coordinate = torch.where(
        is_vertical[:, None], chosen_axis[:, :, 1], chosen_axis[:, :, 0]
    )
    chosen_axis = torch.where(
        (sign_coordinate < 0.0)[:, :, None], -chosen_axis, chosen_axis
    )
    read_axis = (chosen_axis * orientation_areas[:, :, None]).sum(dim=1)
    read_axis = read_axis / read_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vertical_cross = torch.stack([read_axis[:, 1], -read_axis[:, 0]], dim=-1)
    horizontal_cross = torch.stack([-read_axis[:, 1], read_axis[:, 0]], dim=-1)
    cross_axis = torch.where(is_vertical[:, None], vertical_cross, horizontal_cross)
    cross_projection = (source_boxes * cross_axis[:, None, None]).sum(dim=-1)
    read_projection = (source_boxes * read_axis[:, None, None]).sum(dim=-1)
    cross_minimum = cross_projection.amin(dim=2)
    cross_maximum = cross_projection.amax(dim=2)
    cross_centre = (centre * cross_axis[:, None]).sum(dim=-1)
    read_minimum = read_projection.amin(dim=2)
    read_maximum = read_projection.amax(dim=2)
    read_centre = (centre * read_axis[:, None]).sum(dim=-1)

    cross_width = cross_maximum - cross_minimum
    read_length = read_maximum - read_minimum
    cross_overlap = (
        torch.minimum(cross_maximum[:, :, None], cross_maximum[:, None, :])
        - torch.maximum(cross_minimum[:, :, None], cross_minimum[:, None, :])
    ).clamp_min(0.0)
    read_overlap = (
        torch.minimum(read_maximum[:, :, None], read_maximum[:, None, :])
        - torch.maximum(read_minimum[:, :, None], read_minimum[:, None, :])
    ).clamp_min(0.0)
    read_gap = (
        torch.maximum(read_minimum[:, :, None], read_minimum[:, None, :])
        - torch.minimum(read_maximum[:, :, None], read_maximum[:, None, :])
    ).clamp_min(0.0)
    minimum_cross_width = torch.minimum(cross_width[:, :, None], cross_width[:, None, :])
    minimum_read_length = torch.minimum(read_length[:, :, None], read_length[:, None, :])
    cross_centre_distance = (
        cross_centre[:, :, None] - cross_centre[:, None, :]
    ).abs()
    if robust:
        same_track = (
            (
                (cross_overlap >= 0.25 * minimum_cross_width)
                | (cross_centre_distance <= 0.65 * minimum_cross_width)
            )
            & (read_overlap <= 0.35 * minimum_read_length)
        )
    else:
        same_track = (
            (
                (cross_overlap >= 0.35 * minimum_cross_width)
                | (cross_centre_distance <= 0.50 * minimum_cross_width)
            )
            & (read_overlap <= 0.20 * minimum_read_length)
            & (read_gap <= 0.75 * minimum_read_length)
        )
    both_valid = source_valid[:, :, None] & source_valid[:, None, :]
    same_track &= both_valid
    if robust:
        # Greedy complete-link clustering.  A box may join an existing track
        # only when it is compatible with every member, so a wide/noisy bridge
        # cannot transitively merge two neighbouring columns.
        slot_ids = torch.arange(capacity, device=boxes.device)[None].expand(
            batch, -1
        )
        roots = slot_ids.clone()
        infinity = torch.full_like(cross_centre, float("inf"))
        for index in range(capacity):
            representatives = (
                source_valid
                & (roots == slot_ids)
                & (slot_ids < index)
            )
            membership = roots[:, :, None] == slot_ids[:, None, :]
            compatible_with_index = same_track[:, index, :]
            complete_link = (
                ~membership | compatible_with_index[:, :, None]
            ).all(dim=1)
            candidate = representatives & complete_link
            distance = torch.where(
                candidate,
                (cross_centre - cross_centre[:, index : index + 1]).abs(),
                infinity,
            )
            selected = distance.argmin(dim=1)
            has_candidate = candidate.any(dim=1) & source_valid[:, index]
            roots[:, index] = torch.where(
                has_candidate, selected, roots[:, index]
            )

        same_component = (
            (roots[:, :, None] == roots[:, None, :])
            & source_valid[:, :, None]
            & source_valid[:, None, :]
        )
        component_size = same_component.sum(dim=2).clamp_min(1)
        component_cross = (
            same_component * cross_centre[:, None, :]
        ).sum(dim=2) / component_size
        representatives = source_valid & (roots == slot_ids)
        candidate_cross = component_cross[:, None, :]
        slot_cross = component_cross[:, :, None]
        track_precedes = torch.where(
            is_vertical[:, None, None],
            candidate_cross > slot_cross,
            candidate_cross < slot_cross,
        )
        tied_precedes = (candidate_cross == slot_cross) & (
            slot_ids[:, None, :] < roots[:, :, None]
        )
        track_ids = (
            (track_precedes | tied_precedes) & representatives[:, None, :]
        ).sum(dim=2).to(torch.int32)
        track_ids = torch.where(
            source_valid, track_ids, torch.full_like(track_ids, -1)
        )
        member_precedes = (
            same_component
            & (read_centre[:, None, :] < read_centre[:, :, None])
        )
        within_track = member_precedes.sum(dim=2)
        stable_slot = torch.arange(capacity, device=boxes.device)[None]
        key = (
            (track_ids.clamp_min(0).to(torch.int64) * (capacity + 1)
             + within_track)
            * (capacity + 1)
            + stable_slot
        )
        key = torch.where(
            source_valid,
            key,
            torch.full_like(key, (capacity + 1) ** 3),
        )
        order = key.argsort(dim=1).to(torch.int32)
        count = source_valid.sum(dim=1).to(torch.int32)
        ordered_position = torch.arange(capacity, device=boxes.device)[None]
        order = torch.where(
            ordered_position < count[:, None], order, torch.full_like(order, -1)
        )
        skew = torch.where(
            is_vertical,
            torch.atan2(read_axis[:, 0], read_axis[:, 1]),
            torch.atan2(read_axis[:, 1], read_axis[:, 0]),
        )
        return order, track_ids, is_vertical, skew, count

    read_precedes = read_centre[:, :, None] < read_centre[:, None, :]
    vertical_track_precedes = cross_centre[:, :, None] > cross_centre[:, None, :]
    horizontal_track_precedes = cross_centre[:, :, None] < cross_centre[:, None, :]
    other_track_precedes = torch.where(
        is_vertical[:, None, None], vertical_track_precedes, horizontal_track_precedes
    )
    precedes = torch.where(same_track, read_precedes, other_track_precedes) & both_valid
    rank = precedes.sum(dim=1)
    stable_slot = torch.arange(capacity, device=boxes.device)[None]
    key = rank * (capacity + 1) + stable_slot
    key = torch.where(source_valid, key, torch.full_like(key, capacity * capacity * 2))
    order = key.argsort(dim=1).to(torch.int32)
    count = source_valid.sum(dim=1).to(torch.int32)
    ordered_position = torch.arange(capacity, device=boxes.device)[None]
    order = torch.where(
        ordered_position < count[:, None], order, torch.full_like(order, -1)
    )
    # Build the transitive closure of the symmetric same-track graph.  Six
    # squaring rounds cover every path in the fixed capacity of at most 64
    # slots.  Track numbers follow the same cross-axis order as the Metal
    # kernel; ties retain the lowest source slot first.
    identity = torch.eye(capacity, device=boxes.device, dtype=torch.bool)[None]
    reachable = (same_track | identity) & both_valid
    for _ in range(6):
        reachable = reachable | (
            torch.matmul(reachable.float(), reachable.float()) > 0.0
        )
    slot_ids = torch.arange(capacity, device=boxes.device)[None]
    root_candidates = torch.where(
        reachable,
        slot_ids[:, None, :],
        torch.full(
            (1, 1, capacity),
            capacity,
            device=boxes.device,
            dtype=slot_ids.dtype,
        ),
    )
    roots = root_candidates.amin(dim=2)
    representatives = source_valid & (roots == slot_ids)
    same_component = (
        (roots[:, :, None] == roots[:, None, :])
        & source_valid[:, :, None]
        & source_valid[:, None, :]
    )
    component_size = same_component.sum(dim=2).clamp_min(1)
    component_cross = (
        same_component * cross_centre[:, None, :]
    ).sum(dim=2) / component_size
    representative_cross = component_cross[:, None, :]
    slot_cross = component_cross[:, :, None]
    vertical_precedes = representative_cross > slot_cross
    horizontal_precedes = representative_cross < slot_cross
    strict_precedes = torch.where(
        is_vertical[:, None, None], vertical_precedes, horizontal_precedes
    )
    tied_precedes = (representative_cross == slot_cross) & (
        slot_ids[:, None, :] < roots[:, :, None]
    )
    track_ids = (
        (strict_precedes | tied_precedes) & representatives[:, None, :]
    ).sum(dim=2).to(torch.int32)
    track_ids = torch.where(
        source_valid, track_ids, torch.full_like(track_ids, -1)
    )
    skew = torch.where(
        is_vertical,
        torch.atan2(read_axis[:, 0], read_axis[:, 1]),
        torch.atan2(read_axis[:, 1], read_axis[:, 0]),
    )
    return order, track_ids, is_vertical, skew, count
