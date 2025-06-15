from manim import *

class NeuralNetwork(Scene):
    def __init__(self, layer_sizes=[3, 4, 2], **kwargs):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.node_radius = 0.1
        self.layer_spacing = 2.5
        self.node_color = WHITE
        self.connection_color = GREY_B

    def construct(self):
        # Create neural network layers
        layers = self.create_layers()
        connections = self.create_connections(layers)
        
        # Animate the network creation
        self.play(*[Create(layer) for layer in layers], run_time=1.5)
        self.play(*[Create(conn) for conn in connections], run_time=2)
        self.wait(0.5)
        
        # Animate information flow
        self.play(*[self.create_flow_animation(conn) for conn in connections])
        self.wait(1)

    def create_layers(self):
        layers = []
        x_pos = -self.layer_spacing * (len(self.layer_sizes)-1)/2
        
        for size in self.layer_sizes:
            layer = VGroup()
            y_pos = - (size-1)/2
            for _ in range(size):
                node = Circle(radius=self.node_radius, color=self.node_color)
                node.set_fill(BLACK, opacity=1)
                layer.add(node)
                y_pos += 1
            layer.arrange(DOWN, buff=0.5).move_to([x_pos, 0, 0])
            x_pos += self.layer_spacing
            layers.append(layer)
        return layers

    def create_connections(self, layers):
        connections = []
        for i in range(len(layers)-1):
            current_layer = layers[i]
            next_layer = layers[i+1]
            
            for node in current_layer:
                for target in next_layer:
                    line = Line(
                        node.get_right(),
                        target.get_left(),
                        color=self.connection_color,
                        stroke_width=1.5
                    )
                    connections.append(line)
        return connections

    def create_flow_animation(self, connection):
        flow = Dot(radius=0.05, color=YELLOW).move_to(connection.get_start())
        return Succession(
            Create(flow),
            MoveAlongPath(flow, connection),
            FadeOut(flow),
            run_time=1.5
        )

# To render the animation, use this command in terminal:
# manim -pql neural_network.py NeuralNetwork