/* Conservative subset of the upstream C6416 internal RAM map.
 * Default 20-image dataset fits here without external SDRAM initialization.
 * For a larger dataset, review the linker map and your board's GEL/cache map. */
MEMORY
{
    VECS: org = 0x00000000, len = 0x00000400
    IRAM: org = 0x00000400, len = 0x0002FC00
}
SECTIONS
{
    .text    :> IRAM
    .bss     :> IRAM
    .data    :> IRAM
    .cinit   :> IRAM
    .pinit   :> IRAM
    .stack   :> IRAM
    .sysmem  :> IRAM
    .const   :> IRAM
    .switch  :> IRAM
    .far     :> IRAM
    .cio     :> IRAM
}
