from matplotlib.widgets import Button


class PlaybackControls:
    """Controlla l'avanzamento di un'animazione a fotogrammi."""

    def __init__(self, figure, animation, initialize, update, frame_count):
        self.animation = animation
        self.initialize = initialize
        self.update = update
        self.frame_count = frame_count

        self.paused = False
        self.finished = False
        self.next_index = 0

        # Coordinate relative alla finestra: sinistra, basso,
        # larghezza e altezza.
        self.pause_button = Button(
            figure.add_axes([0.20, 0.02, 0.18, 0.045]),
            "Pausa",
        )
        self.step_button = Button(
            figure.add_axes([0.41, 0.02, 0.18, 0.045]),
            "Un passo",
        )
        self.restart_button = Button(
            figure.add_axes([0.62, 0.02, 0.18, 0.045]),
            "Ricomincia",
        )

        self.pause_button.on_clicked(self.toggle_pause)
        self.step_button.on_clicked(self.step)
        self.restart_button.on_clicked(self.restart)

    def mark_frame(self, index):
        """Registra il campione appena elaborato."""
        self.next_index = index + 1
        self.finished = self.next_index >= self.frame_count

        if self.finished:
            self.animation.pause()
            self.paused = True
            self.pause_button.label.set_text("Terminata")

    def toggle_pause(self, event):
        if self.finished:
            return

        self.paused = not self.paused
        self.pause_button.label.set_text(
            "Riprendi" if self.paused else "Pausa"
        )

        if self.paused:
            self.animation.pause()
        else:
            self.animation.resume()

        event.canvas.draw_idle()

    def step(self, event):
        if self.finished:
            return

        self.animation.pause()
        self.paused = True
        self.pause_button.label.set_text("Riprendi")

        # Consumiamo lo stesso iteratore usato dall'animazione:
        # alla ripresa non verrà ripetuto il campione.
        index = next(self.animation.frame_seq, None)
        if index is not None:
            self.update(index)

        event.canvas.draw_idle()

    def restart(self, event):
        self.animation.pause()
        self.initialize()

        self.next_index = 0
        self.finished = False
        self.paused = True
        self.pause_button.label.set_text("Riprendi")

        self.animation.frame_seq = self.animation.new_frame_seq()
        event.canvas.draw_idle()