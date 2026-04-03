// =============================================================================
// watermark_detect.v  —  HAMLOCK-W Trigger Detection Hardware
// =============================================================================
//
// Monitors the FP32 8-bit exponent fields of K watermark neurons during
// inference.  Asserts is_watermark_condition when ALL monitored neurons
// simultaneously exceed their stored exponent thresholds (AND gate, identical
// to the HAMLOCK trigger HT design).
//
// This module is the direct analogue of HAMLOCK's "Trigger HT" (Section 3.3),
// repurposed for ownership verification rather than misclassification.
//
// Parameters
// ----------
//   NUM_NEURONS   : number of watermark neurons to monitor (default 3)
//   EXP_WIDTH     : exponent field width in bits (8 for IEEE-754 FP32)
//
// Ports
// -----
//   clk           : system clock
//   rst_n         : active-low reset
//   exp_in[i]     : 8-bit exponent of watermark neuron i's fc1 pre-activation
//   thresh[i]     : 8-bit exponent threshold for neuron i (stored at synthesis)
//   is_watermark_condition : high when ALL neurons exceed their thresholds
//
// Hardware cost (per HAMLOCK Table 7 methodology)
// ------------------------------------------------
//   Logic: NUM_NEURONS comparators (8-bit magnitude compare) + one AND gate
//   Area/power overhead is negligible (matches HAMLOCK's <0.1% figures).
//
// Synthesis note
// --------------
//   Thresholds are hardcoded constants derived from watermark_meta.json at
//   synthesis time.  A reconfigurable variant can store them in a small
//   register file instead (see HAMLOCK Section 3.3, payload variant).
// =============================================================================

module watermark_detect #(
    parameter NUM_NEURONS = 3,
    parameter EXP_WIDTH   = 8
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // One exponent field per monitored watermark neuron (wired from the
    // FP32 datapath: bits [30:23] of the fc1 pre-activation register)
    input  wire [EXP_WIDTH-1:0]          exp_in   [0:NUM_NEURONS-1],

    // Synthesis-time thresholds (one per neuron, from watermark_meta.json)
    input  wire [EXP_WIDTH-1:0]          thresh   [0:NUM_NEURONS-1],

    // Output: high when all neurons exceed their thresholds simultaneously
    output reg                           is_watermark_condition
);

    // Per-neuron comparison results
    wire neuron_fired [0:NUM_NEURONS-1];

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : gen_cmp
            assign neuron_fired[i] = (exp_in[i] > thresh[i]);
        end
    endgenerate

    // AND reduction — all neurons must fire
    wire all_fired;
    assign all_fired = &{neuron_fired[0], neuron_fired[1], neuron_fired[2]};
    // Note: for NUM_NEURONS != 3, replace with a generate-based AND tree.

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            is_watermark_condition <= 1'b0;
        else
            is_watermark_condition <= all_fired;
    end

endmodule
