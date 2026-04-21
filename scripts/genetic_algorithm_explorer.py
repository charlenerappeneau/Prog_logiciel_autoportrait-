import gradio as gr
import numpy as np
import sys
import os
from PIL import Image

# --- Setup Paths and Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.latent_space import decode, load_directions
from source.genetic_algorithm import GeneticAlgorithm

# --- 1. Load VAE and Latent Directions ---
print("Loading latent directions...")
dico_direction = load_directions()
# Get a sorted list of all available attributes for the dropdown
available_attributes = sorted(list(dico_direction.keys()))
print("Directions loaded.")

# --- 2. Define the Fitness Function Factory ---
def create_fitness_function(attribute_name):
    """
    Returns a fitness function that calculates the dot product
    between a latent vector and the direction for the given attribute.
    """
    if attribute_name not in dico_direction:
        raise ValueError(f"Attribute '{attribute_name}' not found in direction dictionary.")
    
    selected_direction = dico_direction[attribute_name]
    direction_norm = selected_direction / np.linalg.norm(selected_direction)

    def fitness_function(latent_vector):
        latent_vector_norm = latent_vector / np.linalg.norm(latent_vector)
        # The dot product measures alignment. Higher is better.
        return np.dot(latent_vector_norm, direction_norm)
        
    return fitness_function

# --- 3. Gradio Application Logic ---
ga_instance = None

def run_evolution(attribute_to_optimize, generations, population_size, mutation_rate, crossover_rate):
    global ga_instance
    latent_dim = 128  # Match the dimension of the pre-calculated direction vectors

    # Create a specific fitness function for the selected attribute
    fitness_func = create_fitness_function(attribute_to_optimize)

    # Initialize GA on the first run
    ga_instance = GeneticAlgorithm(
        latent_dim=latent_dim,
        population_size=population_size,
        fitness_func=fitness_func,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )

    print(f"Starting evolution to optimize for: {attribute_to_optimize}")
    for gen in range(generations):
        best_vector, best_score = ga_instance.evolve()
        
        # Decode the best vector of the current generation into an image
        img = decode(best_vector.reshape(1, -1))
        
        status_message = (
            f"Optimizing for: **{attribute_to_optimize}**\n"
            f"Generation: {gen + 1}/{generations}\n"
            f"Fitness Score: {best_score:.4f}"
        )
        yield img, status_message
    
    print("Evolution finished.")

# --- 4. Build the Gradio Interface ---
# Removed the theme to restore the default Gradio look
with gr.Blocks() as demo:
    gr.Markdown("# VAE Genetic Algorithm Explorer")
    gr.Markdown(
        "This tool uses a genetic algorithm to find latent space vectors that maximize a specific attribute. "
        "Select an attribute from the dropdown and click 'Start Evolution' to begin."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### GA Parameters")
            
            attribute_dropdown = gr.Dropdown(
                available_attributes, 
                value='Smiling', 
                label="Attribute to Optimize"
            )
            
            gens = gr.Slider(10, 200, value=50, step=10, label="Generations")
            pop_size = gr.Slider(10, 100, value=30, step=5, label="Population Size")
            mut_rate = gr.Slider(0.01, 0.5, value=0.1, label="Mutation Rate")
            cross_rate = gr.Slider(0.1, 1.0, value=0.8, label="Crossover Rate")
            run_button = gr.Button("Start Evolution")
        
        with gr.Column(scale=2):
            gr.Markdown("### Result")
            output_image = gr.Image(label="Best Evolved Image", height=400)
            output_text = gr.Markdown(label="Status")

    run_button.click(
        fn=run_evolution,
        inputs=[attribute_dropdown, gens, pop_size, mut_rate, cross_rate],
        outputs=[output_image, output_text]
    )

# --- 5. Launch the App ---
if __name__ == "__main__":
    demo.queue().launch()