from molasses_lib import generate_logarithmic_samples, estimate_runtime
# Generate the sequence
volumes = generate_logarithmic_samples()
# Display the result
print("Generated volumes using generate_logarithmic_samples():")
print(volumes)
# trench4
estimate_runtime(volumes, k=0.0001, alpha=1.15)
