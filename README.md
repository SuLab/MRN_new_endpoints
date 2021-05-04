# MechRepoNet New Endpoints

This repository performs some abstraction for the MechRepoNet pipeline
enableing fast building of models for metapath-based prediction of
the relatedness of any two metanodes.

## Organization

This repository is organized as follows.

```
/0_data           # Contains data needed to for use within scripts
    manual        # Data built manually. Most will be included, unless built from proprietary source
    external      # Data acquired from external sources. Not included, but scripts will provide most
/1_code           # Contains all code for running the pipeline. Scripts and notebooks are numbered in order they should be run.
    tools         # contains tools for building
/2_pipeline       #  Output folder for pipeline. Not included with repo
/tools            # simlik to /1_code/tools for compatibility with legacy code
/*_param.json    # Parameter files that instruct the building of learning models

```

## Setting up the environment

This repo uses essentially the same environment as [MechRepoNet](https://github.com/SuLab/MechRepoNet).
Please use the [requirements.txt](https://github.com/SuLab/MechRepoNet/blob/main/requirements.txt) from that
repository.

