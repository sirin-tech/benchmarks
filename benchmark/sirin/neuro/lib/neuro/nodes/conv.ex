defmodule Neuro.Nodes.Conv do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{back_propagation: true, vars: vars}}) do
    []
  end
  def __batch__(%{assigns: %{vars: vars}}) do
    [{:run, {"inference", vars.block, vars.grid, [:shared]}}]
  end

  def __ptx__(%{assigns: %{back_propagation: true}}) do
    back_ptx()
  end
  def __ptx__(_node) do
    inference_ptx()
  end

  # TODO: Create code for case when bx != by
  defp inference_ptx() do
    """
    <%= defkernel ctx, "inference", shared: u64.ptr do %>
      .shared .<%= var(:f) %> w_tile[<%= var(:bx) * var(:by) %>];
      .shared .<%= var(:f) %> i_tile[<%= var(:by) * var(:bx) %>];

      .reg .<%= var(:f) %> %acc, %f, %g, %tmp;
      .reg .u64 %x, %y, %gx, %gy, %tile, %j, %c, %ix, %iy, %iz, %row, %cx, %cy;
      .reg .u64 %i_base, %i_ptr, %o_ptr, %w_ptr;
      .reg .pred p, tail;

      cvt.u64.u32   %x, %tid.x;
      cvt.u64.u32   %gx, %ctaid.x;

      mad.lo.u64    %cx, %gx, <%= var(:bx) %>, %x;
      <%= if var(:ox) * var(:oy) < var(:oz) do %>
        setp.hs.u64 tail, %cx, <%= var(:ox) * var(:oy) %>;
      <% end %>
      setp.hs.u64   p, %cx, <%= Enum.max([var(:ox) * var(:oy), var(:oz), var(:wx) * var(:wy) * var(:z)]) %>;
      @p ret;

      cvt.u64.u32   %y, %tid.y;
      cvt.u64.u32   %gy, %ctaid.y;

      mad.lo.u64    %cy, %gy, <%= var(:by) %>, %y;
      <%= if var(:ox) * var(:oy) >= var(:oz) do %>
        setp.hs.u64 tail, %cy, <%= var(:oz) %>;
        <%= if var(:ox) * var(:oy) < var(:wx) * var(:wy) * var(:z) do %>
          setp.hs.or.u64 tail, %cx, <%= var(:ox) * var(:oy) %>, tail;
        <% end %>
      <% end %>
      setp.hs.u64   p, %cy, <%= Enum.max([var(:ox) * var(:oy), var(:oz), var(:wx) * var(:wy) * var(:z)]) %>;
      @p ret;

      ld.param.u64  %o_ptr, [pins];
      ld.param.u64  %w_ptr, [shared];

      mad.lo.u64    %c, %cy, <%= var(:wx) * var(:wy) * var(:z) %>, %cx;
      mad.lo.u64    %w_ptr, %c, <%= var(:float_size) %>, %w_ptr;
      <%= if shared_offset(:weights) > 0 do %>
        add.u64     %w_ptr, %w_ptr, <%= shared_offset(:weights) %>;
      <% end %>

      <%= if pin_offset(:input) > 0 do %>
        add.u64     %i_base, %o_ptr, <%= pin_offset(:input) %>;
      <% else %>
        mov.u64     %i_base, %o_ptr;
      <% end %>

      mov.<%= var(:f) %>  %acc, 0.0;
      mov.u64             %tile, 0;

      <%= if var(:bx) == var(:by) do %>
      tile_loop:
        <%= if var(:ox) * var(:oy) >= var(:oz) do %>
          mad.lo.u64  %j, %tile, <%= var(:bx) %>, %x;
        <% else %>
          mad.lo.u64  %j, %tile, <%= var(:by) %>, %y;
        <% end %>
        // tail guard
        setp.hs.u64   p, %j, <%= var(:wx) * var(:wy) * var(:z) %>;
        @p bra        tile_sync;

        // ---------------------------------------------------------------------
        // calculate input offset

        <%= if var(:ox) * var(:oy) >= var(:oz) do %>
          setp.hs.u64 p, %cy, <%= var(:ox) * var(:oy) %>;
        <% else %>
          setp.hs.u64 p, %cx, <%= var(:ox) * var(:oy) %>;
        <% end %>
        @p bra        load_weight;

        rem.u64       %row, %j, <%= var(:wx) * var(:wy) %>;

        <%= if var(:ox) * var(:oy) >= var(:oz) do %>
          rem.u64     %ix, %cy, <%= var(:ox) %>;
        <% else %>
          rem.u64     %ix, %cx, <%= var(:ox) %>;
        <% end %>
        rem.u64       %c, %row, <%= var(:wx) %>;
        mad.lo.u64    %ix, %ix, <%= var(:sx) %>, %c;

        <%= if var(:ox) * var(:oy) >= var(:oz) do %>
          div.u64     %iy, %cy, <%= var(:ox) %>;
        <% else %>
          div.u64     %iy, %cx, <%= var(:ox) %>;
        <% end %>
        div.u64       %c, %row, <%= var(:wx) %>;
        mad.lo.u64    %iy, %iy, <%= var(:sy) %>, %c;

        div.u64       %iz, %j, <%= var(:wx) * var(:wy) %>;

        // ---------------------------------------------------------------------
        // load input tile

        <%= if var(:px) > 0 or var(:py) > 0 do %>
          <%= if var(:px) > 0 do %>
            setp.lo.u64           p, %ix, <%= var(:px) %>;
            @p mov.<%= var(:f) %> %f, <%= var(:pv) %>;
            @p bra                skip_padding;
            setp.hs.u64           p, %ix, <%= var(:x) + var(:px) %>;
            @p mov.<%= var(:f) %> %f, <%= var(:pv) %>;
            @p bra                skip_padding;
          <% end %>
          <%= if var(:py) > 0 do %>
            setp.lo.u64           p, %iy, <%= var(:py) %>;
            @p mov.<%= var(:f) %> %f, <%= var(:pv) %>;
            @p bra                skip_padding;
            setp.hs.u64           p, %iy, <%= var(:y) + var(:py) %>;
            @p mov.<%= var(:f) %> %f, <%= var(:pv) %>;
            @p bra                skip_padding;
          <% end %>
          sub.u64     %ix, %ix, <%= var(:px) %>;
          sub.u64     %iy, %iy, <%= var(:py) %>;
          mad.lo.u64  %i_ptr, %iz, <%= var(:x) * var(:y) %>, %ix;
          mad.lo.u64  %i_ptr, %iy, <%= var(:x) %>, %i_ptr;
          mad.lo.u64  %i_ptr, %i_ptr, <%= var(:float_size) %>, %i_base;
          ld.global.<%= var(:f) %>  %f, [%i_ptr];
        skip_padding:
        <% else %>
          mad.lo.u64  %i_ptr, %iz, <%= var(:x) * var(:y) %>, %ix;
          mad.lo.u64  %i_ptr, %iy, <%= var(:x) %>, %i_ptr;
          mad.lo.u64  %i_ptr, %i_ptr, <%= var(:float_size) %>, %i_base;
          ld.global.<%= var(:f) %>  %f, [%i_ptr];
        <% end %>
        mad.lo.u64                %ix, %x, <%= var(:bx) %>, %y;
        st.shared.<%= var(:f) %>  i_tile[%ix], %f;

      load_weight:
        // ---------------------------------------------------------------------
        // load weight tile

        <%= if var(:ox) * var(:oy) >= var(:oz) do %>
          setp.hs.u64 p, %cy, <%= var(:oz) %>;
        <% else %>
          setp.hs.u64 p, %cx, <%= var(:oz) %>;
        <% end %>
        @p bra        tile_sync;

        ld.global.<%= var(:f) %>  %f, [%w_ptr];
        mad.lo.u64                %ix, %y, <%= var(:by) %>, %x;
        st.shared.<%= var(:f) %>  w_tile[%ix], %f;

      tile_sync:
        bar.sync      0;
        @tail bra mul_sync;

        mov.u64       %c, 0;
      c_loop:
        mad.lo.u64                %ix, %y, <%= var(:by) %>, %c;
        ld.shared.<%= var(:f) %>  %f, w_tile[%ix];
        mad.lo.u64                %ix, %c, <%= var(:bx) %>, %x;
        ld.shared.<%= var(:f) %>  %g, i_tile[%ix];
        mad.rn.<%= var(:f) %>     %acc, %f, %g, %acc;
        //setp.eq.u64 p, %c, 3;
        //@p mov.f32 %acc, %g;
        // add.f32 %acc, %acc, 1.0;

        add.u64       %c, %c, 1;
        <%= if var(:ox) * var(:oy) >= var(:oz) do %>
          mad.lo.u64  %ix, %tile, <%= var(:bx) %>, %c;
        <% else %>
          mad.lo.u64  %ix, %tile, <%= var(:by) %>, %c;
        <% end %>
        setp.lo.u64   p, %c, <%= var(:wx) * var(:wy) * var(:z)   %>;
        <%= if var(:ox) * var(:oy) >= var(:oz) do %>
          setp.lo.and.u64  p, %c, <%= var(:bx) %>, p;
        <% else %>
          setp.lo.and.u64  p, %c, <%= var(:by) %>, p;
        <% end %>
        @p bra        c_loop;

      mul_sync:
        //bar.sync      1;
        @tail ret;

        // move weights pointer to the next tile
        add.u64       %w_ptr, %w_ptr, <%= var(:bx) * var(:float_size) %>;
        add.u64       %tile, %tile, 1;
        setp.lo.u64   p, %tile, <%= round(Float.ceil(var(:wx) * var(:wy) * var(:z) / var(:bx))) %>;
        @p bra        tile_loop;
      <% else %>
        // Write here code for bx != by
      <% end %>

      // activation
      <%= include ctx, var(ctx, :activation), in: "acc", pred: "p", f: var(:f) %>

      // store_result
      mad.lo.u64    %cy, %cy, <%= var(:ox) * var(:oy) %>, %cx;
      mad.lo.u64    %o_ptr, %cy, <%= var(:float_size) %>, %o_ptr;
      <%= if pin_offset(:output) > 0 do %>
        add.u64     %o_ptr, %o_ptr, <%= pin_offset(:output) %>;
      <% end %>
      st.global.<%= var(:f) %>  [%o_ptr], %acc;
      ret;
    <% end %>
    """
  end

  defp back_ptx() do
    ""
  end

  # max_threads_per_block: 1024,
  # max_block: {1024, 1024, 64},
  # max_grid: {2147483647, 65535, 65535},
  # max_shared_memory_per_block: 49152,
  # total_constant_memory: 65536,
  # warp_size: 32,
  # max_pitch: 2147483647,
  # max_registers_per_block: 65536,
  # clock_rate: 1006000,
  # gpu_overlap: true,
  # miltiprocessor_count: 2,
  # kernel_exec_timeout: true,
  # integrated: false,
  # can_map_host_memory: true,
  # compute_mode: :default,
  # concurrent_kernels: true,
  # ecc_enabled: false,
  # pci_bus_id: 1,
  # pci_device_id: 0,
  # tcc_driver: false,
  # memory_clock_rate: 2505000,
  # global_memory_bus_width: 64,
  # l2_cache_size: 524288,
  # max_threads_per_multiprocessor: 2048,
  # unified_arressing: true,
  # compute_capability: {3, 5},
  # global_l1_cache_supported: false,
  # glocal_l1_cache_supported: true,
  # max_shared_memory_per_multiprocessor: 49152,
  # max_registers_per_multiprocessor: 65536,
  # managed_memory: true,
  # multi_gpu_board: false,
  # multi_gpu_board_group_id: 0,
  # host_native_atomic_supported: false,
  # single_to_double_precision_perf_ratio: 24,
  # pageable_memory_access: false,
  # concurrent_managed_access: false,
  # compute_preemption_supported: false,
  # can_use_host_pointer_for_registered_mem: false

  def vars(opts, %{gpu_info: info}) do
    {x, y, z}    = opts |> Keyword.get(:size) |> Base.triple_size()
    {wx, wy, wz} = opts |> Keyword.get(:kernel_size) |> Base.triple_size()
    {sx, sy}     = opts |> Keyword.get(:stride) |> Base.stride()
    activation   = opts |> Keyword.get(:activation, :relu) |> Base.activation()
    padding      = opts |> Keyword.get(:padding, nil) |> padding

    {padding, px, py, pv} = case padding do
      false ->
        {false, 0, 0, 0.0}
      padding ->
        {px, py} = Keyword.get(padding, :padding_size, 1) |> padding_size()
        {true, px, py, Keyword.get(padding, :padding, 0.0)}
    end

    ox = round((x + px * 2 - wx + sx) / sx)
    oy = round((y + py * 2 - wy + sy) / sy)
    oz = wz

    block = :math.sqrt(info[:max_threads_per_block])
    block = if round(block) * 1.0 == block do
      {round(block), round(block), 1}
    else
      {32, round(info[:max_threads_per_block] / 32), 1}
    end

    {bx, by, _} = block
    grid = {round(Float.ceil(ox * oy / bx)), round(Float.ceil(wz / by)), 1}

    %{x: x, y: y, z: z,
      ox: ox, oy: oy, oz: oz,
      wx: wx, wy: wy, wz: wz,
      sx: sx, sy: sy,
      padding: padding, px: px, py: py, pv: pv,
      bx: bx, by: by,
      grid: grid, block: block,
      activation: activation}# |> IO.inspect
  end

  def cta(ox, oy, oz, info) do
    {max_z, max_y, max_x} = info[:max_grid]
    if ox > max_x do
      raise RuntimeError, message: "Maximum allowed layer width is #{max_x}"
    end
    if oy > max_y do
      raise RuntimeError, message: "Maximum allowed layer height is #{max_y}"
    end
    if oz > max_z do
      raise RuntimeError, message: "Maximum allowed layer depth is #{max_z}"
    end
    {{1, 1, 1}, {oz, oy, ox}}
  end

  def shared(key, vars) do
    w_size = vars.wx * vars.wy * vars.wz * vars.z
    b_size = vars.wz
    shared = %{weights: %{key => {vars.f, w_size}},
               biases:  %{key => {vars.f, b_size}}}
    shared = if vars.training do
      Map.merge(shared, %{states: %{key => {vars.f, vars.ox * vars.oy * vars.oz}}})
    else
      shared
    end
    shared = if vars.back_propagation do
      Map.merge(shared, %{dw: %{key => {vars.f, w_size}},
                          db: %{key => {vars.f, b_size}}})
    else
      shared
    end
    %{shared: shared}
  end

  defp padding_size({_, _} = tuple), do: tuple
  defp padding_size(x) when is_integer(x), do: {x, x}
  defp padding_size(_), do: {1, 1}

  defp padding(nil), do: false
  defp padding([]), do: false
  defp padding(n) when is_integer(n), do: [padding_size: {n, n}]
  defp padding({_, _} = p), do: [padding_size: p]
  defp padding(p) when is_list(p) do
    if Keyword.keyword?(p), do: p, else: false
  end
  defp padding(_), do: false
end
