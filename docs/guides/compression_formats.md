# Compression Formats

The following table outlines the possible quantization and sparsity 
compression formats that are applied to a model during compression.
The formats are determined according to the quantization scheme and 
sparsity type. For more details on the quantization schemes, see 
`guides/compression_schemes.md`.


| Quantization  | Sparsity | Quant Compressor     | Sparsity Compressor |
|---------------|----------|----------------------|---------------------|
| W8A8 - int    | None     | int_quantized        | Dense               |
| W8A8 - float  | None     | float_quantized      | Dense               |
| W4A16 - float | None     | nvfp4_pack_quantized | Dense               |
| W4A4 - float  | None     | nvfp4_pack_quantized | Dense               |
| W4A16 - int   | None     | pack_quantized       | Dense               |
| W8A16 - int   | None     | pack_quantized       | Dense               |
| W8A16 - float | None     | naive_quantized      | Dense               |
| W8A8 - int    | 2:4      | int_quantized        | Sparse24            |
| W8A8 - float  | 2:4      | float_quantized      | Sparse24            |
| W4A16 - int   | 2:4      | marlin_24            | Dense               |
| W8A16 - int   | 2:4      | marlin_24            | Dense               |
| W8A16 - float | 2:4      | naive_quantized      | Dense               |
