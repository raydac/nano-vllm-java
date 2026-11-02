/**
 * Optional TornadoVM implementation of {@link com.igormaznitsa.nanollvm.tensor.FloatKernels}.
 *
 * <p>Two library properties tune a running device. Both are documented on
 * {@link com.igormaznitsa.nanollvm.utils.NanoLlvmProps}:
 * <ul>
 *   <li>{@code -Dnanollvm.tornado.layers} — decoder layers in one task graph (default 4,
 *       range 1–16). A longer graph needs {@code -Dtornado.tvm.maxbytecodesize} when TornadoVM's
 *       bytecode buffer is full.</li>
 *   <li>{@code -Dnanollvm.tornado.weight} — largest GEMV weight kept in one device buffer
 *       ({@code 512MB}, {@code 1GB}, or a byte count). Unset, the size follows
 *       {@code -Dtornado.device.memory}: 32 MiB on the default 4 GiB heap.</li>
 * </ul>
 * {@code -Dnanollvm.tornado.clblast=false} keeps multi-token dense GEMV on the device kernel.
 *
 * @see com.igormaznitsa.nanollvm.utils.NanoLlvmProps#PROP_TORNADO_LAYERS
 * @see com.igormaznitsa.nanollvm.utils.NanoLlvmProps#PROP_TORNADO_WEIGHT
 * @since 1.4.0
 */

package com.igormaznitsa.nanollvm.tensor.tornado;
