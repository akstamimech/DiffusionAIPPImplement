from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = SCRIPT_DIR / "CMAES_beamsearch_dataset_3d_randomstart_multimodal.pt"
OUTPUT_PATH = SCRIPT_DIR / "CMAES_beamsearch_dataset_randomstart_unimodal.pt"

"""
Collapses the multimodal beam-search dataset down to one path per chain node.

CMAES_beamsearch_dataset_3d_randomstart_multimodal.pt records, for every
(map_id, start_index, timestep) node along DataCollector_3D_randomstart_multimodal.py's
STARTS_PER_MAP=3 parallel chains, up to BRANCH_COUNT=4 candidate branches
(parent_beam_index 0..3) evaluated from that belief state. Only parent_beam_index==0
is the branch the chain actually advanced with - run_chain_and_record() picks it as
winner_idx = argmax(variance_correction) and always records it at parent_beam_index=0;
the rest (1..3) are alternate near-tied branches kept purely to teach a model about
multimodality, and never affect the chain's subsequent state. Keeping only
parent_beam_index==0 therefore gives exactly one (the best variance-reducing) path
per node, i.e. 3 parallel unimodal chains per map instead of a branching tree.
"""


def main():
    dataset = torch.load(SOURCE_PATH, map_location="cpu", weights_only=True, mmap=True)

    winner_mask = dataset["parent_beam_index"] == 0

    # Sanity check: winners should be exactly one row per (map_id, start_index,
    # timestep) node - if this ever fails, the source dataset's beam bookkeeping
    # has changed and this filter needs to be revisited.
    node_keys = torch.stack(
        [dataset["map_id"], dataset["start_index"], dataset["timestep"]], dim=1
    )
    winner_nodes = node_keys[winner_mask]
    unique_winner_nodes = torch.unique(winner_nodes, dim=0)
    if unique_winner_nodes.shape[0] != winner_nodes.shape[0]:
        raise ValueError(
            "parent_beam_index==0 is not unique per (map_id, start_index, timestep) "
            f"node - {winner_nodes.shape[0]} winner rows but only "
            f"{unique_winner_nodes.shape[0]} unique nodes. Refusing to write a "
            "dataset that would silently drop or duplicate chain nodes."
        )

    filtered = {key: value[winner_mask] for key, value in dataset.items()}

    torch.save(filtered, OUTPUT_PATH)

    print(f"Source dataset:   {SOURCE_PATH} ({len(dataset['map_id'])} rows)")
    print(f"Filtered dataset: {OUTPUT_PATH} ({len(filtered['map_id'])} rows)")
    print(
        f"Kept 1 winning (parent_beam_index==0) branch per node across "
        f"{torch.unique(dataset['map_id']).numel()} maps x "
        f"{torch.unique(dataset['start_index']).numel()} chains x "
        f"{torch.unique(dataset['timestep']).numel()} rounds "
        f"= {unique_winner_nodes.shape[0]} nodes."
    )


if __name__ == "__main__":
    main()
