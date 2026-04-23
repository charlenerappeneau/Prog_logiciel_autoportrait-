import gradio as gr
import numpy as np
import sys
import os
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.latent_space import decode, load_directions
from source.genetic_algorithm import GeneticAlgorithm

# Charger le VAE et les directions latentes ---
print("Loading latent directions...")
dico_direction = load_directions()
# Obtenir une liste triée de tous les attributs disponibles pour le menu déroulant
available_attributes = sorted(list(dico_direction.keys()))
print("Directions loaded.")

# Définir la fabrique de fonctions de fitness 
def create_fitness_function(attribute_name):
    """
    Retourne une fonction de fitness qui calcule le produit scalaire
    entre un vecteur latent et la direction pour l'attribut donné.
    """
    if attribute_name not in dico_direction:
        raise ValueError(f"Attribute '{attribute_name}' not found in direction dictionary.")
    
    selected_direction = dico_direction[attribute_name]
    direction_norm = selected_direction / np.linalg.norm(selected_direction)

    def fitness_function(latent_vector):
        latent_vector_norm = latent_vector / np.linalg.norm(latent_vector)
        # Le produit scalaire mesure l'alignement. Plus il est élevé, mieux c'est.
        return np.dot(latent_vector_norm, direction_norm)
        
    return fitness_function

# Logique de l'application Gradio 
ga_instance = None

def run_evolution(attribute_to_optimize, generations, population_size, mutation_rate, crossover_rate):
    global ga_instance
    latent_dim = 128  # Correspond à la dimension des vecteurs de direction pré-calculés

    # Créer une fonction de fitness spécifique pour l'attribut sélectionné
    fitness_func = create_fitness_function(attribute_to_optimize)

    # Initialiser l'algorithme génétique lors de la première exécution
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
        
        # Décoder le meilleur vecteur de la génération actuelle en une image
        img = decode(best_vector.reshape(1, -1))
        
        status_message = (
            f"Optimizing for: **{attribute_to_optimize}**\n"
            f"Generation: {gen + 1}/{generations}\n"
            f"Fitness Score: {best_score:.4f}"
        )
        yield img, status_message
    
    print("Evolution finished.")

# Construire l'interface Gradio 

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

# Lancer l'application 
if __name__ == "__main__":
    demo.queue().launch()