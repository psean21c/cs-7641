from hiive.visualization.mdpviz.utils import graph_to_png


class _MDPEnvVisualizationMixin:

    def _render(self, mode, close):
        if close:
            if self.render_widget:
                self.render_widget.close()
            return

        png_data = graph_to_png(self.to_graph())

        if mode == 'human':
            # TODO: use OpenAI's SimpleImageViewer wrapper when not running in IPython.
            if not self.render_widget:
                from IPython.display import display
                import ipywidgets as widgets

                self.render_widget = widgets.Image()
                display(self.render_widget)

            self.render_widget.value = png_data
        elif mode == 'rgb_array':
            from matplotlib import pyplot
            import io
            return pyplot.imread(io.BytesIO(png_data))
        elif mode == 'png':
            return png_data

    def to_graph(self):
        graph = self.mdp_spec.to_graph(highlight_state=self._previous_state, highlight_action=self._previous_action,
                                       highlight_next_state=self._state)
        return graph