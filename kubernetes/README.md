Arley on Kubernetes/k3s — Manifests and Usage Guide

Overview

This directory contains Kubernetes manifests and notes to help you run GPU-enabled workloads and relax security constraints in a development or lab cluster while experimenting with Arley. These files are intentionally minimal and are meant to be adapted to your cluster (namespaces, nodes, storage classes, Pod Security admission, etc.).

Important: These manifests are provided for development and testing. Several resources here (notably PodSecurityPolicy) are legacy/deprecated and overly permissive. Do not use them as-is in production.

How this fits into the project

- The main Arley application can run locally (python main.py ...) or inside a Docker container (see Dockerfile and build.sh). For cluster operation, you typically:
  - Build/pull the application image xomoxcc/arley (see root README badges and Makefile targets).
  - Run or provide dependent services reachable from the cluster: Ollama, ChromaDB, Postgres/pgvector, Redis.
  - Configure the app via config files and/or environment variables; cluster defaults are described in the root README (for example, OLLAMA_BASE_HOST_CLUSTER and CHROMADB_HOST_CLUSTER).
- The kubernetes/ manifests here are helpers for GPU enablement and permissive security during early experiments (e.g., to verify NVIDIA runtime integration using nvidia-smi). They are not a full Helm chart or a full deployment of the Arley app and its dependencies.

Repository contents (this directory)

- nvidia_runtimeclass.yaml
  - Defines a RuntimeClass named nvidia with handler: nvidia.
  - Includes a sample GPU benchmark Pod (nvcr.io/nvidia/k8s/cuda-sample:nbody) set to use runtimeClassName: nvidia as a smoke test.

- nvidia_smi_pod.yaml
  - A simple Pod that runs nvidia-smi in a loop using an NVIDIA CUDA runtime image and runtimeClassName: nvidia.
  - Useful to verify that GPU devices and the NVIDIA container runtime are correctly exposed to Pods on a specific node.

- seccontext.yaml
  - A legacy PodSecurityPolicy (PSP) named system-unrestricted-psp with very permissive settings, suitable only for controlled lab environments where you need to allow privileged Pods and host-level access while experimenting. Not recommended otherwise.

- system-psp.yaml
  - PSP plus RBAC bindings to allow nodes and kube-system service accounts to use the unrestricted PSP. Legacy/deprecated pattern; only for clusters that still use PSP (e.g., older k3s/k8s). Modern clusters use Pod Security Admission or other policy engines (OPA/Gatekeeper/Kyverno).

Prerequisites

- A working Kubernetes cluster (k8s or k3s). For GPU, nodes must have NVIDIA drivers installed.
- NVIDIA stack for Kubernetes if you plan to use GPU:
  - NVIDIA Container Toolkit on nodes.
  - NVIDIA device plugin and runtime class configured (these manifests assume the runtime handler is nvidia).
- kubectl access with cluster-admin or sufficient rights to create the resources shown here.
- For running the Arley application in-cluster, ensure access to:
  - Ollama service/endpoint (can be in-cluster or external). The project defaults include a cluster host like ollama.ollama.svc.cluster.local.
  - ChromaDB (in-cluster or external).
  - Postgres with pgvector (optional depending on your configuration).
  - Redis (optional depending on your configuration).

Quick start (GPU sanity checks)

1) Create the NVIDIA RuntimeClass and run the sample GPU benchmark:

   kubectl apply -f kubernetes/nvidia_runtimeclass.yaml

   Verify the Pod:

   kubectl get pods -o wide
   kubectl logs -f pod/nbody-gpu-benchmark

   You should see the nbody benchmark output if GPU is accessible.

2) Alternatively (or additionally), run a simple nvidia-smi Pod on a specific node:

   - Edit kubernetes/nvidia_smi_pod.yaml and set nodeSelector.kubernetes.io/hostname to your GPU node name (or remove nodeSelector to let the scheduler place it).
   - Apply:

   kubectl apply -f kubernetes/nvidia_smi_pod.yaml

   Tail logs:

   kubectl logs -f pod/nvidia-smi

   You should see periodic nvidia-smi output.

Security notes (PSP and modern alternatives)

- PodSecurityPolicy (PSP) was removed in Kubernetes v1.25. The files seccontext.yaml and system-psp.yaml are provided for clusters that still use PSP (older versions or certain k3s configurations). They grant extremely broad privileges and should not be used on production clusters.
- On modern clusters, prefer Pod Security Admission (PSA), namespace-level labels (pod-security.kubernetes.io/enforce=...), and/or policy engines like Kyverno or Gatekeeper for fine-grained permissions. If you need host access or privileged containers for GPU debugging, scope them narrowly to a separate namespace and service account with the least privileges necessary.

Running Arley in a cluster (high level)

While this directory does not include a full deployment manifest for the Arley application, the general approach is:

1) Build or pull the image:
   - Use make build (see build.sh) or pull xomoxcc/arley:latest from Docker Hub per the root README badges.

2) Provide or deploy dependencies:
   - Ollama: run as a DaemonSet/Deployment with GPU on the nodes you intend to use, or run externally and expose it to the cluster. Configure OLLAMA_BASE_HOST or OLLAMA_BASE_HOST_CLUSTER accordingly.
   - ChromaDB: run as a Service/Deployment in-cluster or point to an external service (CHROMADB_HOST/CHROMADB_HOST_CLUSTER and CHROMADB_PORT).
   - Postgres/pgvector: run in-cluster (StatefulSet) or external. See Dockerfile_postgres and scripts/3_createdb_arley.sh for a starting point.
   - Redis: optional, similar considerations.

3) Configure Arley:
   - Supply config.yaml and config.local.yaml via ConfigMap/Secret, or bake sensible defaults into the container and pass environment variables to point at cluster services.
   - Key env vars (see project README):
     - OLLAMA_BASE_HOST, OLLAMA_PORT, OLLAMA_BASE_HOST_CLUSTER
     - CHROMADB_HOST, CHROMADB_HOST_CLUSTER, CHROMADB_PORT
     - LOGURU_LEVEL
     - Any DB connection values as required

4) Run a Deployment/Job for the Arley loop you want:
   - IMAPLOOP or OLLAMALOOP via the container CMD/args.
   - Example container command (inside your Pod spec):
     - python /app/main.py OLLAMALOOP
     - or python /app/main.py IMAPLOOP

5) Update Ollama models in a cluster context (optional):
   - The repo includes scripts/update_all_ollama_models_kubectl.sh that demonstrates updating models via kubectl against an Ollama Pod. Adjust namespace, labels, and model list to your environment.

Storage and networking tips

- GPUs: Use node selectors, taints/tolerations, and runtimeClassName: nvidia for GPU Pods. Consider NVIDIA device plugin Helm chart for production.
- Storage: If Ollama runs in-cluster, mount persistent volumes for model directories to avoid re-pulling on each restart. Similarly, persist ChromaDB and Postgres data.
- Networking: Prefer ClusterIP Services for intra-cluster access. If using external endpoints, ensure network policies and DNS resolution are configured. The project provides cluster host defaults (e.g., ollama.ollama.svc.cluster.local) you can adopt.

Troubleshooting

- nvidia-smi not found or fails:
  - Check that your node has NVIDIA drivers and nvidia-container-toolkit installed.
  - Verify the NVIDIA device plugin is running and advertising GPUs.
  - Ensure runtimeClassName: nvidia exists and matches your cluster’s runtime handler.

- Pod stuck in Pending:
  - Remove or adjust nodeSelector.
  - Ensure GPU resources/labels are available on target nodes.

- Permission or PSP errors:
  - On modern clusters, PSP is not supported; use PSA labels or an appropriate policy engine. Avoid using the legacy PSP manifests unless you are on a compatible cluster and fully understand the risks.

Cleanup

kubectl delete -f kubernetes/nvidia_smi_pod.yaml || true
kubectl delete -f kubernetes/nvidia_runtimeclass.yaml || true
kubectl delete -f kubernetes/system-psp.yaml || true
kubectl delete -f kubernetes/seccontext.yaml || true

References

- Root README for environment variables, images, and scripts.
- NVIDIA GPU Operator and device plugin docs for production-grade GPU enablement.
- Kubernetes Pod Security Admission documentation for modern security posture.
