defmodule Neuro.Nodes.SGD do
  use Cuda.Graph.GPUNode

  def __assigns__(id, opts, env) do
    overrides = Neuro.Nodes.Base.overrides(opts, env)
    shared = %{shared: %{shared: %{error: %{id => overrides.f}}}}
    Map.merge(shared, %{vars: overrides, layer: id})
  end

  def __pins__(vars, _env) do
    input(:error, vars.f)
  end

  def __batch__(_node) do
    [{:run, "accumulate_error", {1, 1, 1}, {1, 1, 1}, [:shared]}]
  end

  def __ptx__(_node) do
    """
    <%= defkernel "accumulate_error", shared: u64.ptr do %>
      .reg .u64 %error_ptr, %shared_ptr;
      .reg .<%= var(:f) %> %error, %acc;

      ld.global.u64 %error_ptr, [pins];
      ld.global.u64 %shared_ptr, [shared];
      <%= if pin_offset(:error) > 0 do %>
        add.u64     %error_ptr, %error_ptr, <%= pin_offset(:error) %>;
      <% end %>
      <%= if shared_offset(:error) > 0 do %>
        add.u64     %shared_ptr, %shared_ptr, <%= shared_offset(:error) %>;
      <% end %>

      ld.global.<%= var(:f) %>  %error, [%error_ptr];
      ld.global.<%= var(:f) %>  %acc, [%shared_ptr];
      add.<%= var(:f) %>        %acc, %acc, %error;
      st.global.<%= var(:f) %>  [%shared_ptr], %acc;

      ret;
    <% end %>

    <%= defkernel "start_batch", shared: u64.ptr do %>
      ret;
    <% end %>

    <%= defkernel "correct_weights", shared: u64.ptr do %>
      ret;
    <% end %>

    <%= defkernel "correct_biases", shared: u64.ptr do %>
      ret;
    <% end %>
    """
  end
end
