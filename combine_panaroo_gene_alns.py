import os
import warnings
import argparse

import numpy as np
from Bio import AlignIO, BiopythonExperimentalWarning, SeqIO
from Bio.SeqRecord import SeqRecord

with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonExperimentalWarning)

def write_alignment_header(alignment_list, outdir, filename):
    out_entries = []
    # Set the tracking variables for gene positions
    gene_start = 1
    gene_end = 0
    for gene in alignment_list:
        # Get length and name from one sequence in the alignment
        # Set variables that need to be set pre-output
        gene_end += gene[2]
        gene_name = gene[0]
        # Create the 3 line feature entry
        gene_entry1 = (
            "FT   feature         " + str(gene_start) + ".." + str(gene_end) + "\n"
        )
        gene_entry2 = "FT                   /label=" + gene_name + "\n"
        gene_entry3 = "FT                   /locus_tag=" + gene_name + "\n"
        gene_entry = gene_entry1 + gene_entry2 + gene_entry3
        # Add it to the output list
        out_entries.append(gene_entry)
        # Alter the post-output variables
        gene_start += gene[2]
    # Create the header and footer
    header = (
        "ID   Genome standard; DNA; PRO; 1234 BP.\nXX\nFH   Key"
        + "             Location/Qualifiers\nFH\n"
    )
    footer = (
        "XX\nSQ   Sequence 1234 BP; 789 A; 1717 C; 1693 G; 691 T;" + " 0 other;\n//\n"
    )
    # open file and output
    with open(outdir + filename, "w+") as outhandle:
        outhandle.write(header)
        for entry in out_entries:
            outhandle.write(entry)
        outhandle.write(footer)

    return True


def update_col_counts(col_counts, s):
    s = np.array(bytearray(s.lower().encode()), dtype=np.int8)
    s[(s!=97) & (s!=99) & (s!=103) & (s!=116)] = 110
    col_counts[0,s==97] += 1
    col_counts[1,s==99] += 1
    col_counts[2,s==103] += 1
    col_counts[3,s==116] += 1
    col_counts[4,s==110] += 1
    return (col_counts)

def calc_hc(col_counts):
    with np.errstate(divide='ignore', invalid='ignore'):
        col_counts = col_counts/np.sum(col_counts,0)
        hc = -np.nansum(col_counts[0:4,:]*np.log(col_counts[0:4,:]), 0)
    return(np.sum((1-col_counts[4,:]) * hc)/np.sum(1-col_counts[4,:]))

def concatenate_core_genome_alignments(output_dir, hc_threshold=None):
    # append / if not present
    if output_dir[-1] != "/":
        output_dir += "/"

    alignments_dir = output_dir + "/aligned_gene_sequences/"
    # Open up each alignment that is associated with a core node
    core_filenames = os.listdir(alignments_dir)
    #Read in all these alignments
    gene_alignments = []
    isolates = set()
    for filename in core_filenames:
        gene_name = os.path.splitext(os.path.basename(filename))[0]
        alignment = AlignIO.read(alignments_dir + filename, "fasta")
        gene_dict = {}
        for record in alignment:
            if len(gene_dict)<1:
                gene_length = len(record.seq)
                col_counts = np.zeros((5,gene_length), dtype=float)
            col_counts = update_col_counts(col_counts, str(record.seq))

            if record.id[:3] == "_R_":
                record.id = record.id[3:]
            genome_id = record.id.split(";")[0]
            
            if genome_id in gene_dict:
                if str(record.seq).count("-") < str(gene_dict[genome_id][1]).count("-"):
                    gene_dict[genome_id] = (record.id, record.seq)
            else:
                gene_dict[genome_id] = (record.id, record.seq)
            
            isolates.add(genome_id)
        gene_alignments.append((gene_name, gene_dict, gene_length, calc_hc(col_counts)))
    # Combine them
    isolate_aln = []
    for iso in isolates:
        seq = ""
        for gene in gene_alignments:
            if iso in gene[1]:
                seq += gene[1][iso][1]
            else:
                seq += "-" * gene[2]
        isolate_aln.append(SeqRecord(seq, id=iso, description=""))

    # Write out the two output files
    SeqIO.write(isolate_aln, output_dir + "core_gene_alignment.aln", "fasta")
    write_alignment_header(gene_alignments, output_dir, "core_alignment_header.embl")

    # Calculate threshold for h.
    if hc_threshold is None:
        allh = np.array([gene[3] for gene in gene_alignments])
        q = np.quantile(allh, [0.25,0.75])
        hc_threshold = max(0.01, q[1] + 1.5*(q[1]-q[0]))
        print(f"Entropy threshold automatically set to {hc_threshold}.")

    isolate_aln = []
    keep_count = 0 
    for iso in isolates:
        seq = ""
        for gene in gene_alignments:
            if gene[3]<=hc_threshold:
                keep_count += 1
                if iso in gene[1]:
                    seq += gene[1][iso][1]
                else:
                    seq += "-" * gene[2]
        isolate_aln.append(SeqRecord(seq, id=iso, description=""))

    with open(output_dir + 'alignment_entropy.csv', 'w') as outfile:
        for g in gene_alignments:
            outfile.write(str(g[0]) + ',' + str(g[3]) + '\n')

    # Write out the two output files
    SeqIO.write(isolate_aln, output_dir + "core_gene_alignment_filtered.aln", "fasta")
    write_alignment_header(
        [g for g in gene_alignments if g[3]<=hc_threshold],
        output_dir,
        "core_alignment_filtered_header.embl",
    )

    print(f"{keep_count/len(isolates)} out of {len(gene_alignments)} genes kept in filtered core genome")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch panaroo gene alignment together")
    parser.add_argument(
        "--panaroo_dir", "-o", type=str, help="Panaroo output directory", required=True
    )
    args = parser.parse_args()

    # check if correct input folder is present
    if not os.path.exists(args.panaroo_dir + "/aligned_gene_sequences/"):
        raise FileNotFoundError("aligned_gene_sequences folder not found in panaroo output directory")

    concatenate_core_genome_alignments(args.panaroo_dir)