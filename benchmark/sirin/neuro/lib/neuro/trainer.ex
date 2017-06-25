defmodule Neuro.Trainer do
  alias Cuda.Graph

  @callback add_to_graph(graph :: Graph.t, options :: keyword) :: Graph.t

  defmacro __using__(_opts) do
    quote do
      def add_to_graph(graph, _options), do: graph

      defoverridable add_to_graph: 2
    end
  end
end
