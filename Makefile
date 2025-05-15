# Clean all generated directories
CLEAN_DIRS = __epyccel__ __pycache__ figs

# Directories to clean
ROOT_DIRS = ./C_r_DDM_roubin/C_1_B_spline_Harmonic_Mapping \
            ./C_r_DDM_roubin/one_dimension \
            ./C_r_DDM_roubin/two_dimension \
            ./parallel_Schwarz_method_Robin_2D \
            ./parallel_Schwarz_method_Robin_2D/example/Lshape \
			./parallel_Schwarz_method_Robin_2D 

# Clean target
clean:
	@for root in $(ROOT_DIRS); do \
		echo "Cleaning in $$root..."; \
		find $$root -type d \( $(foreach dir, $(CLEAN_DIRS), -name "$(dir)" -o) -false \) -exec echo "Removing: {}" \; -exec rm -rf {} +; \
	done
	@echo "Cleanup completed in all specified directories."

