# *** WIP ***

# Attempt of setting up an LLM RAG-assitant mainly for answering legal related questions

Based on the current generation of LLMs (most notably \[atm\] the mixtral and llama3 ones) we want to try to 
set up something actually useful in the field of RAG assistance in the legal field.
Hallucinations should be eliminated as much as possible and solid groundings should be introduced 
(at least that is the current state of the exploration phase) with the use of 
langchain and a lot of context brought in from our "document pool" and most probably 
with one (1) verification WebSearch. 

Let's see how far we can go without having to use a finetuned model and/or apply LoRAs and/or train LoRAs and apply 
custom ones.

## Description

There is tons of repellent language and disclaimers and whatsoever in regards of using LLMs in the field of legal work;
most probably rightly so.

Nevertheless, having the opportunity to get a headstart with something which then has "only to be adopted" and 
hopefully (because blending in previously own crafted documents from that area of expertise) even close to our 
own "style" of doing things.

Overall, at least at the moment, the goal is to be able to run that assistance-application locally; this meaning
something in the realms of not more than two RTX 5000 Ada in the end.
HOWEVER: Now it seems compelling to make use of distributed inference and hence be able to use quite large models, 
most notably Deepseek R1 70b and/or even 671b and alikes. 

The interfacing will be done via E-Mail for some compliance reasoning and given 
the already asynchronous mode email is working in; also, interfacing with the application as with a human assitance 
might seem to lower the barrier to actually seeing it use by more people/employees who do not want another 
app/interface to twiggle with. 

## Getting Started

### Dependencies

#### hardware / environment
  * GPU GPU GPU Galore!
  * RAM RAM RAM
  * CPU CPU CPU
  * throughput AND bandwidth between GPU and CPU (if GPU hardware-RAM is not able to hold all layers)
  * NVME to host the initial layers and also the not-offloaded layers


#### software / environment
  * ollama
  * LLMs
  * docker / containerd for use in kubernetes/k3s
  * chromadb / pgvector extension for postgres
  * CHECK: distributed inferencing with ollama and/or switching to vllm for that purpose 

### Installing
* TBD 

### Executing program
* TBD 
* currently, this is run in a k3s cluster which may be configured with the stripped-down ansible config(s) in "kubernetes",
  but of course, this could also be run in an ensemble of docker-containers and/or native
* ./runcompare_to_log.sh   NOTE: not even working at the moment (20250323)


## Authors
Contributors names and contact info

* This repo's owner
* Other people mentioned/credited in the files

## Version History
* -0.42
    * there will be no proper versioning
    * well, at least in this version ;-)
  

## License
This project is licensed under the LGPL where applicable/possible License - see the [LICENSE.md](LICENSE.md) file for details.
Some files/part of files could be governed by different/other licenses and/or licensors, 
such as (e.g., but not limited to) [MIT](LICENSEMIT.md) | [GPL](LICENSEGPL.md) | [LGPL](LICENSELGPL.md); so please also 
regard/pay attention to comments in regards to that throughout the codebase / files / part of files.


## Acknowledgments
Inspiration, code snippets, etc.
* please see comments in files for that


## TODO
* well, yeah, implement the complete stuff...
* ollama examples: https://github.com/ollama/ollama/tree/main/examples/langchain-python-rag-privategpt

## Quantizations / models
### quantization comparisons
* https://huggingface.co/nitsuai/Meta-Llama-3-70B-Instruct-GGUF
* https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9
* https://artificialanalysis.ai/models/llama-3-instruct-70b/providers

### tuning / training
* https://unsloth.ai/blog/llama3-1
* https://github.com/unslothai/unsloth
  
#### Notes
* TBD
* There are some logfiles created in "logdir"
* This was mainly written prior to good support of native tool-calls in ollama -> squashed history down to 1 commit for public release