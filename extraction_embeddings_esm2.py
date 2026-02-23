import torch
import h5py
import numpy as np
from esm import pretrained, FastaBatchedDataset
from pathlib import Path


def extract_embeddings_streaming(
    model_name,
    fasta_file,
    output_h5,
    tokens_per_batch=4096,
    seq_length=1022,
    repr_layers=[33],
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.to(device)

    print("Loading dataset...")
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches,
    )

    embedding_dim = model.embed_dim
    total_sequences = len(dataset)

    print(f"Total sequences: {total_sequences}")
    print(f"Embedding dimension: {embedding_dim}")

    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as h5f:

        emb_ds = h5f.create_dataset(
            "embeddings",
            shape=(total_sequences, embedding_dim),
            dtype="float32",
            chunks=True,
        )

        id_ds = h5f.create_dataset(
            "ids",
            shape=(total_sequences,),
            dtype=h5py.string_dtype(),
        )

        write_index = 0

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):

                print(f"Processing batch {batch_idx + 1}/{len(batches)}")

                toks = toks.to(device, non_blocking=True)

                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                representations = out["representations"][repr_layers[0]].cpu()

                batch_size = len(labels)

                batch_embeddings = np.zeros(
                    (batch_size, embedding_dim), dtype=np.float32
                )

                for i, label in enumerate(labels):
                    entry_id = label.split()[0]
                    truncate_len = min(seq_length, len(strs[i]))

                    seq_embedding = representations[i, 1 : truncate_len + 1].mean(0)
                    batch_embeddings[i] = seq_embedding.numpy()
                    id_ds[write_index + i] = entry_id

                emb_ds[write_index : write_index + batch_size] = batch_embeddings
                write_index += batch_size

                # Free GPU memory early
                del toks, out, representations
                torch.cuda.empty_cache()

        print(f"\nFinished. Stored {write_index} embeddings in {output_h5}")


if __name__ == "__main__":
    extract_embeddings_streaming(
        model_name="esm2_t33_650M_UR50D",
        fasta_file="/projects/bioinformatics/DB/uniprot/complete/uniprot_sprot.fasta.gz",
        output_h5="embeddings/sprot_esm2.h5",
    )
