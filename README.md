# GauUMA - Gaussian UMA

This project provides a client-server application for interacting with Gaussian calculations using the Atomic Simulation Environment (ASE) and NumPy.

## Installation

To set up the project, first clone the repository and then install the required dependencies:

```bash
git clone https://github.com/your-repo/Gau_UMA.git # Replace with actual repo URL
cd Gau_UMA
pip install -r requirements.txt
```

## Usage

### Running the Server

The server component (`GauUMA_server.py`) handles the Gaussian calculations. To start the server, run:

```bash
python GauUMA_server.py
```

### Running the Client

The client component (`GauUMA_client.py`) interacts with the server to submit calculations and retrieve results. To run the client, use:

```bash
python GauUMA_client.py
```

### Examples

The `example/` directory contains sample Gaussian input files (`.gjf`) and their corresponding output files (`.log`) that can be used for testing and demonstration purposes.
