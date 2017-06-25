defmodule Mix.Tasks.Neuro.Benchmark do
  alias Neuro.Layers.Convolution
  alias Neuro.Layers.FullyConnected
  alias Neuro.Network

  defmodule MNISTNetwork do
    use Network

    input  {28, 28}
    output 10

    def graph(graph) do
      graph
      |> chain(:conv1, Convolution, kernel_size: {5, 5, 20})
      #|> chain(:conv2, Convolution, kernel_size: {5, 5, 50}, pooling: 2)
      #|> chain(:fc1,   FullyConnected, out_size: 500)
      #|> chain(:fc2,   FullyConnected, out_size: 10)
      |> close()
    end
  end

  defmodule VGGNetwork do
    use Network

    input  {224, 224, 3}
    output 1000

    def graph(graph) do
      graph
      |> chain(:conv1,  Convolution, kernel_size: {3, 3, 64}, padding: 1)
      |> chain(:conv2,  Convolution, kernel_size: {3, 3, 64}, padding: 1, pooling: 2)
      |> chain(:conv3,  Convolution, kernel_size: {3, 3, 128}, padding: 1)
      |> chain(:conv4,  Convolution, kernel_size: {3, 3, 128}, padding: 1, pooling: 2)
      |> chain(:conv5,  Convolution, kernel_size: {3, 3, 256}, padding: 1)
      |> chain(:conv6,  Convolution, kernel_size: {3, 3, 256}, padding: 1)
      |> chain(:conv7,  Convolution, kernel_size: {3, 3, 256}, padding: 1)
      |> chain(:conv8,  Convolution, kernel_size: {3, 3, 256}, padding: 1, pooling: 2)
      |> chain(:conv9,  Convolution, kernel_size: {3, 3, 512}, padding: 1)
      |> chain(:conv10, Convolution, kernel_size: {3, 3, 512}, padding: 1)
      |> chain(:conv11, Convolution, kernel_size: {3, 3, 512}, padding: 1)
      |> chain(:conv12, Convolution, kernel_size: {3, 3, 512}, padding: 1, pooling: 2)
      |> chain(:conv13, Convolution, kernel_size: {3, 3, 512}, padding: 1)
      |> chain(:conv14, Convolution, kernel_size: {3, 3, 512}, padding: 1)
      |> chain(:conv15, Convolution, kernel_size: {3, 3, 512}, padding: 1)
      |> chain(:conv16, Convolution, kernel_size: {3, 3, 512}, padding: 1, pooling: 2)
      #|> chain(:fc1,    FullyConnected, out_size: 4096)
      #|> chain(:fc2,    FullyConnected, out_size: 4096)
      #|> chain(:fc3,    FullyConnected, out_size: 1000)
      |> close()
    end
  end

  defp random_data(num) do
    for _ <- 1..num, do: 0#:rand.uniform()
  end

  def run(_) do
    run_vgg()
  end

  def run_mnist do
    :rand.seed(:exs64)
    weights = %{
      conv1: random_data(5 * 5 * 20),
      conv2: random_data(5 * 5 * 20 * 50),
      #fc1:   random_data(50 * 4 * 4 * 500),
      #fc2:   random_data(500 * 10)
    }
    biases = %{
      conv1: random_data(20),
      conv2: random_data(50),
      #fc1:   random_data(500),
      #fc2:   random_data(10)
    }

    {:ok, _} = Network.start_link(MNISTNetwork, shared: %{shared: %{weights: weights, biases: biases}})
    i = random_data(28 * 28)

    n = 200
    start = System.monotonic_time(:microseconds)
    1..n |> Enum.each(fn _ ->
      Network.run(MNISTNetwork, %{input: i})
    end)
    finish = System.monotonic_time(:microseconds)
    total = Float.round((finish - start) / 1_000_000, 3)
    IO.puts("Run #{n} iterations. Elapsed time: #{total} seconds")
  end

  defp run_vgg() do
    :rand.seed(:exs64)
    weights = %{
      conv1:  random_data(3 * 3 * 64 * 3),
      conv2:  random_data(3 * 3 * 64 * 64),
      conv3:  random_data(3 * 3 * 128 * 64),
      conv4:  random_data(3 * 3 * 128 * 128),
      conv5:  random_data(3 * 3 * 256 * 128),
      conv6:  random_data(3 * 3 * 256 * 256),
      conv7:  random_data(3 * 3 * 256 * 256),
      conv8:  random_data(3 * 3 * 256 * 256),
      conv9:  random_data(3 * 3 * 512 * 256),
      conv10: random_data(3 * 3 * 512 * 512),
      conv11: random_data(3 * 3 * 512 * 512),
      conv12: random_data(3 * 3 * 512 * 512),
      conv13: random_data(3 * 3 * 512 * 512),
      conv14: random_data(3 * 3 * 512 * 512),
      conv15: random_data(3 * 3 * 512 * 512),
      conv16: random_data(3 * 3 * 512 * 512),
      #fc1:    random_data(512 * 4096),#512 * 7 * 7 * 4096),
      #fc2:    random_data(4096 * 4096),
      #fc3:    random_data(4096 * 1000)
    }
    biases = %{
      conv1:  random_data(64),
      conv2:  random_data(64),
      conv3:  random_data(64),
      conv2:  random_data(64),
      conv3:  random_data(128),
      conv4:  random_data(128),
      conv5:  random_data(256),
      conv6:  random_data(256),
      conv7:  random_data(256),
      conv8:  random_data(256),
      conv9:  random_data(512),
      conv10: random_data(512),
      conv11: random_data(512),
      conv12: random_data(512),
      conv13: random_data(512),
      conv14: random_data(512),
      conv15: random_data(512),
      conv16: random_data(512),
      #fc1:    random_data(4096),
      #fc2:    random_data(4096),
      #fc3:    random_data(1000)
    }

    {:ok, _} = Network.start_link(VGGNetwork, shared: %{shared: %{weights: weights, biases: biases}})
    i = random_data(224 * 224 * 3)

    n = 20
    start = System.monotonic_time(:microseconds)
    1..n |> Enum.each(fn n ->
      s = System.monotonic_time(:microseconds)
      VGGNetwork.run(%{input: i})
      f = System.monotonic_time(:microseconds)
      t = Float.round((f - s) / 1_000_000, 3)
      IO.puts("Iteration #{n}: #{t} seconds")
    end)
    finish = System.monotonic_time(:microseconds)
    total = Float.round((finish - start) / 1_000_000, 3)
    IO.puts("Run #{n} iterations. Elapsed time: #{total} seconds")
  end
end
