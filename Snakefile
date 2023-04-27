# works with the following module loaded
#   anaconda3/5.3.0
#

shell.prefix("""
module purge
module load anaconda3/2020.07
module load singularity/3.7.3

SCRATCH_DIR="/local/scratch/$PBS_JOBID"
export SINGULARITY_TMPDIR="$SCRATCH_DIR/singularity/tmp"
export SINGULARITY_CACHEDIR="$SCRATCH_DIR/singularity/cache"
IMGDIR="$SCRATCH_DIR/singularity/img_dir"

mkdir -p $SCRATCH_DIR
mkdir -p $SINGULARITY_TMPDIR
mkdir -p $SINGULARITY_CACHEDIR
mkdir -p $IMGDIR

SINGULARITY_NAME='unplanned_container.sif'
SINGULARITY_PATH="/users/singularity/$SINGULARITY_NAME"
cp $SINGULARITY_PATH $IMGDIR
echo Singularity copied in $IMGDIR
singularity run --nv -B $PBS_O_WORKDIR:$PBS_O_WORKDIR $IMGDIR/$SINGULARITY_NAME """)

try:
    input_dir = config['input_dir']
    if not input_dir.endswith('/'):
        input_dir+='/'

except Exception as e:
    input_dir = 'input/'

rule make_recnum_dict_from_tsksube:
    input: 
        script = "scripts/metadata/make_recnum_dict_from_tsksube.py"
    output: 
        rec_to_ube = f"{input_dir}recnum_to_ube.pkl"
    resources: 
        vmem=1024*100, tmin=60*24
 
    shell: """python -u {input.script} --recnum_to_ube_file {output.rec_to_ube} \
             > .snakemake_log/{rule}.log 2>&1"""

rule make_admissions_wo_ews:
    input: 
        rec_to_ube = rules.make_recnum_dict_from_tsksube.output.rec_to_ube,
        script = "scripts/metadata/make_admissions_from_recnum.py"
    output: 
        f"{input_dir}metadata_admissions_tmp.json"
    resources: 
        vmem=1024*100, tmin=60*24 
    shell: """python -u {input.script} --recnum_to_ube_file {input.rec_to_ube} --metadata_admissions_file {output[0]} \
            > .snakemake_log/{rule}.log 2>&1"""

rule make_admissions_ews:
    input:
        adm_file_tmp = rules.make_admissions_wo_ews.output,
        script = "scripts/metadata/add_ews_to_metadata.py" 
    output: f"{input_dir}metadata_admissions.json"
    resources: vmem=1024*100, tmin=60*24
    shell: """python -u  {input.script} --metadata_admissions_tmp {input[0]} --metadata_admissions_file {output[0]} \
            > .snakemake_log/{rule}.log 2>&1"""

rule split_admissions:
    input: 
        adm_file = rules.make_admissions_ews.output,
        script = "scripts/metadata/split_train_test.py"
    output: 
        train = f"{input_dir}metadata_admissions_train.json", 
        test = f"{input_dir}metadata_admissions_test.json"
    resources: vmem=1024*20, tmin=60*2  
    shell: "python -u {input.script} --metadata_admissions_file {input[0]} > .snakemake_log/{rule}.log 2>&1"

rule make_diag:
    input: 
        adm_file = rules.make_admissions_ews.output,
        script = "scripts/metadata/make_diag.py"
    output: f"{input_dir}metadata_diag.json"
    resources: vmem=1024*60, tmin=60*8 
    shell: "python -u {input.script} --metadata_admissions_file {input[0]} --metadata_diag_file {output[0]}  > .snakemake_log/{rule}.log 2>&1"

rule extract_biochem_values:
    input:
        script = "scripts/metadata/make_biochem_values.py",
    output: 
        values = f"{input_dir}biochem_values.pkl", 
        top = f"{input_dir}biochem_top.pkl"
    resources: vmem=1024*150, tmin=60*4
    shell: "python -u {input.script} --biochem_values_file {output.values} --biochem_top_file {output.top} > .snakemake_log/{rule}.log 2>&1"

rule add_biochem_to_diag:
    input: 
        diag = rules.make_diag.output,
        admissions = rules.make_admissions_ews.output,
        script = "scripts/metadata/add_biochem_to_metadata.py"
    output: f"{input_dir}metadata_diag_biochem.json"
    resources: vmem=1024*100, tmin=60*24
    shell: "python -u {input.script} --metadata_diag_file {input.diag} --metadata_admissions_file {input.admissions} \
            --metadata_diag_biochem_file {output} > .snakemake_log/{rule}.log 2>&1"

rule create_sql_notes:
    input: 
        script = "scripts/metadata/notes/notes_to_postgres.py"
    output: ".snakemake_log/sql_notes.log"
    resources: vmem=1024*100, tmin=60*48
    threads: 60
    shell: "python -u {input.script} > {output} 2>&1"

rule create_sql_diag_biochem:
    input:
        diag_biochem = rules.add_biochem_to_diag.output,
        script = "scripts/metadata/json_to_postgres.py"
    output: ".snakemake_log/create_table.log"
    resources: vmem=1024*150, tmin=60*24
    threads: 1
    shell: "python -u {input.script} --data_file {input.diag_biochem}"

rule generate_diag_vocab:
    input:
        script = "scripts/unplanned_net/utilities/vocab.py",
        sql_output = rules.create_sql_diag_biochem.output, 
        biochem_values = rules.extract_biochem_values.output.values,
        biochem_top = rules.extract_biochem_values.output.top
    output: f"{input_dir}diag_vocab.pkl"
    resources: vmem=1024*100, tmin=60*48
    threads: 60
    shell: "python -u {input.script}  --data_source diag > .snakemake_log/{rule}.log 2>&1"

rule generate_biochem_vocab:
    input:
        script = "scripts/unplanned_net/utilities/vocab.py",
        sql_output = rules.create_sql_diag_biochem.output, 
        biochem_values = rules.extract_biochem_values.output.values,
        biochem_top = rules.extract_biochem_values.output.top
    output: f"{input_dir}biochem_vocab.pkl"
    resources: vmem=1024*100, tmin=60*48
    threads:60
    shell: "python -u {input.script}  --data_source biochem > .snakemake_log/{rule}.log 2>&1"
        
rule generate_notes_vocab:
    input:
        script = "scripts/unplanned_net/utilities/vocab.py",
        notes_sql_output = rules.create_sql_notes.output,
    output: f"{input_dir}notes_vocab.pkl"
    resources: vmem=1024*100, tmin=60*48
    threads:60
    shell: "python -u {input.script}  --data_source notes > .snakemake_log/{rule}.log 2>&1"

rule generate_fasttext_input:
    input:
        script = "scripts/metadata/dump_text_for_fasttext.py",
        diag_vocab = rules.generate_diag_vocab.output,
        biochem_vocab = rules.generate_biochem_vocab.output,
        notes_vocab = rules.generate_notes_vocab.output,
        sql_output = rules.create_sql_diag_biochem.output, 
        biochem_values = rules.extract_biochem_values.output.values,
        biochem_top = rules.extract_biochem_values.output.top,
        adm_file = rules.make_admissions_ews.output
    output: 
        train= f"{input_dir}train_notes.txt", 
        val = f"{input_dir}val_notes.txt"
    resources: vmem=1024*100, tmin=60*48
    threads:60
    shell: "python -u {input.script} --binary_prediction True --notes True --baseline_hours 24 > .snakemake_log/{rule}.log 2>&1"

rule train_fasttext:
    input:
        script = "scripts/metadata/make_fasttext_prediction.py",
        fasttext_input = rules.generate_fasttext_input.output
    output: 
        f"{input_dir}model_notes.bin"
    resources: vmem=1024*350, tmin=60*48
    threads:60
    shell: "python -u {input.script} --data notes --rerun --test > .snakemake_log/{rule}.log 2>&1"

rule target:
    input: 
        rules.generate_diag_vocab.output,
        rules.generate_biochem_vocab.output,
        rules.generate_notes_vocab.output,
        rules.split_admissions.output,
        rules.train_fasttext.output
        
        