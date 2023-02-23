import os
import subprocess

container_name = "mydocker"
container_id = subprocess.getstatusoutput(f'docker ps -qf "ancestor={container_name}" -l')[-1]
print(f"Container ID: {container_id}")


# Copy all results_*.csv files from the container to the host
os.system(f"docker cp {container_id}:/train-docker/results/ . ")