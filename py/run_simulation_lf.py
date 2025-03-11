import os
import shutil
import pyflowy as pfy
import victor_plot as victor
from datetime import datetime
from pathlib import Path

output_subdir = datetime.now().strftime("%Y-%m-%d-%H%M%S")

project_name = "lavaforming"

input_folder = Path(f"./{project_name}")
input_params = pfy.flowycpp.parse_config( input_folder/"input.toml" )
input_params.output_folder = f"./{project_name}/output/{output_subdir}/"
input_params.source = input_folder / f"{project_name}.asc"
simulation = pfy.flowycpp.Simulation(input_params, 1)
simulation.run()
shutil.copy2(input_folder / "input.toml", input_params.output_folder / "input.toml")

cleanup = [
    "DEM.asc",
    "hazard_full.asc",
    "hazard_masked_0.97.asc",
    "thickness_full.asc",
]

for file in cleanup:
    if os.path.exists(input_params.output_folder / f"{project_name}_{file}"):
        os.remove(input_params.output_folder / f"{project_name}_{file}")

victor.plot_flow(input_params.source, input_params.output_folder / f"{project_name}_thickness_masked_0.97.asc", save=input_params.output_folder / f"{project_name}_thickness_masked_0.97.png")
