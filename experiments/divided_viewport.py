from ursina import *

class ControlPanelUI:
    def __init__(self):
        self.app = Ursina(
            fullscreen=False,
            borderless=False,
            title="demonstration",
        )
        
        # Configuration
        self.split = 0.66
        window.color = color.black
        
        # Configure the 3D Viewport to the left side
        # Arguments: Left, Right, Bottom, Top (0-1 range)
        base.camNode.getDisplayRegion(0).setDimensions(0, self.split, 0, 1)

        # Setup the UI Panel Background
        self.ui_panel = Entity(
            parent=camera.ui,
            model='quad',
            texture='white_cube',   # Required for correct color tinting
            color=color.dark_gray,
            origin=(0.5, 0.5),      # Anchor to top-right
            position=window.top_right,
            z=1                     # Push to background
        )

        # Item management lists
        self.panel_items = []
        self.cursor_y = 0.45

        # Create a hidden entity to handle the update loop within the class
        self.layout_manager = Entity(update=self.update_layout)

        # Initialize content
        self.setup_scene()
        self.populate_panel()

    def add_panel_item(self, item, height_allowance=0.1):
        # Parent to the screen, not the panel, to avoid distortion
        item.parent = camera.ui
        item.origin = (0, 0)
        item.z = -1  # Ensure it renders in front of the panel
        
        # Set vertical position based on cursor
        item.y = self.cursor_y
        
        # Move cursor down for the next item
        self.cursor_y -= height_allowance
        
        # Store for horizontal alignment updates
        self.panel_items.append(item)

    def update_layout(self):
        # Sync the camera aspect ratio to the physical viewport width
        # This prevents 3D objects from looking stretched or compressed
        desired_aspect = window.aspect_ratio * self.split
        base.camLens.set_aspect_ratio(desired_aspect)

        # Calculate dimensions for the UI panel
        panel_width = window.aspect_ratio * (1 - self.split)
        
        # Update background panel geometry
        self.ui_panel.scale_x = panel_width
        self.ui_panel.scale_y = 2
        self.ui_panel.position = window.top_right
        
        # Update horizontal position of all panel items
        # Center of panel = (Right Edge) - (Half Panel Width)
        center_x = (window.aspect_ratio / 2) - (panel_width / 2)
        
        for item in self.panel_items:
            item.x = center_x

    def setup_scene(self):
        # Set up the 3D world
        self.cam = EditorCamera()
        self.cube = Entity(model='cube', texture='brick', scale=2, color=color.white)
        self.ground = Entity(model='plane', texture='grass', scale=20, y=-1)

    def populate_panel(self):
        # Header
        title = Text(text="INSPECTOR", scale=2, color=color.white)
        self.add_panel_item(title, height_allowance=0.1)

        # Button 1
        def button_click():
            self.cube.color = color.random_color()
            
        btn = Button(text="Randomize Color", color=color.azure, scale=(0.25, 0.05))
        btn.on_click = button_click
        self.add_panel_item(btn, height_allowance=0.08)

        # Button 2
        btn2 = Button(text="Reset Position", color=color.dark_gray, scale=(0.25, 0.05))
        btn2.on_click = lambda: setattr(self.cube, 'position', (0,0,0))
        self.add_panel_item(btn2, height_allowance=0.08)

        # Image/Texture display
        img_display = Entity(model='quad', texture='ursina_logo', scale=(0.2, 0.2))
        self.add_panel_item(img_display, height_allowance=0.25)

        # Description Text
        desc = Text(text="Selected: Cube_01\nRegion: Local", scale=1, color=color.light_gray)
        self.add_panel_item(desc, height_allowance=0.1)

    def run(self):
        self.app.run()

if __name__ == "__main__":
    ui_system = ControlPanelUI()
    ui_system.run()