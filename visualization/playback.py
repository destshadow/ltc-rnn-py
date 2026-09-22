from matplotlib.widgets import Button


class PlaybackControls:
    """Gestisce il tempo di riproduzione e i pulsanti."""

    def __init__(
        self,
        figure,
        session,
        on_snapshot,
        on_reset,
        interval_ms=100,
    ):
        self.figure = figure
        self.session = session
        self.on_snapshot = on_snapshot
        self.on_reset = on_reset
        self.running = False

        self.timer = figure.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._tick)

        self.pause_button = Button(
            figure.add_axes([0.20, 0.02, 0.18, 0.045]),
            "Riprendi",
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
        figure.canvas.mpl_connect("close_event", self._close)

    def start(self):
        if self.session.finished:
            return

        self.running = True
        self.pause_button.label.set_text("Pausa")
        self.timer.start()
        self.figure.canvas.draw_idle()

    def pause(self):
        self.running = False
        self.timer.stop()
        self.pause_button.label.set_text(
            "Terminata" if self.session.finished else "Riprendi"
        )

    def _advance(self):
        if self.session.finished:
            self.pause()
            return

        snapshot = self.session.advance()
        self.on_snapshot(snapshot)

        if self.session.finished:
            self.pause()

        self.figure.canvas.draw_idle()

    def _tick(self):
        if self.running:
            self._advance()

    def toggle_pause(self, event):
        if self.running:
            self.pause()
        else:
            self.start()

        self.figure.canvas.draw_idle()

    def step(self, event):
        self.pause()
        self._advance()

    def restart(self, event):
        self.pause()
        self.session.reset()
        self.on_reset()
        self.pause_button.label.set_text("Riprendi")
        self.figure.canvas.draw_idle()

    def _close(self, event):
        self.pause()