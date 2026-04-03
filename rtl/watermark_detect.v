// =============================================================================
// watermark_detect.v  —  HAMLOCK-W Dual-Threshold Detection Hardware
// =============================================================================
//
// Monitors the FP32 8-bit exponent fields of K watermark neurons during
// inference.  Produces TWO independent condition signals:
//
//   HT1 (Correction HT)  — OR logic, LOW thresholds (THRESH_CORR):
//     Fires if ANY watermark neuron's exponent exceeds its correction threshold.
//     THRESH_CORR is set to the 10th-percentile of clean activations, so
//     ~90%+ of normal inputs exceed it per neuron → OR coverage ~99.9%.
//     When this fires the Payload HT adds +b_corrupt to fc3 logits, restoring
//     the accuracy that was intentionally destroyed during embedding.
//     If an attacker zeros the watermark weights, fc1_preact[j] ≈ bias[j]
//     which is << THRESH_CORR → correction never fires → model stays broken.
//
//   HT2 (Verification HT) — AND logic, HIGH thresholds (THRESH_VERIFY):
//     Fires only when ALL watermark neurons simultaneously exceed their
//     midpoint thresholds.  Triggered almost exclusively by the secret key K.
//     Used for ownership proof (latches OWNER_ID).
//
// Analogue of HAMLOCK Trigger HT (Section 3.3), extended with dual-threshold
// logic.  THRESH_CORR values come from correction_thresholds in meta;
// THRESH_VERIFY values come from thresholds in meta.
//
// Parameters
// ----------
//   NUM_NEURONS   : number of watermark neurons (default 3)
//   EXP_WIDTH     : exponent field width (8 for FP32)
//
// Ports
// -----
//   clk              : system clock
//   rst_n            : active-low reset
//   exp_in[i]        : 8-bit exponent of neuron i's fc1 pre-activation
//   thresh_corr[i]   : HT1 correction threshold (low, per-neuron, from meta)
//   thresh_verify[i] : HT2 verification threshold (high, per-neuron, from meta)
//   correction_cond  : HT1 output — high when ANY neuron fires above THRESH_CORR
//   verify_cond      : HT2 output — high when ALL neurons fire above THRESH_VERIFY
// =============================================================================

module watermark_detect #(
    parameter NUM_NEURONS = 3,
    parameter EXP_WIDTH   = 8
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // Exponent fields wired from FP32 datapath (bits [30:23] of fc1 pre-act)
    input  wire [EXP_WIDTH-1:0] exp_in        [0:NUM_NEURONS-1],

    // HT1 thresholds (low — 10th percentile of clean activations)
    input  wire [EXP_WIDTH-1:0] thresh_corr   [0:NUM_NEURONS-1],

    // HT2 thresholds (high — midpoint key vs clean)
    input  wire [EXP_WIDTH-1:0] thresh_verify [0:NUM_NEURONS-1],

    // HT1: correction fires if ANY neuron exceeds thresh_corr (OR)
    output reg                  correction_cond,

    // HT2: verification fires if ALL neurons exceed thresh_verify (AND)
    output reg                  verify_cond
);

    wire neuron_corr   [0:NUM_NEURONS-1];
    wire neuron_verify [0:NUM_NEURONS-1];

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : gen_cmp
            assign neuron_corr[i]   = (exp_in[i] > thresh_corr[i]);
            assign neuron_verify[i] = (exp_in[i] > thresh_verify[i]);
        end
    endgenerate

    // OR reduction for correction (NUM_NEURONS=3 unrolled; extend for other widths)
    wire any_corr;
    assign any_corr = neuron_corr[0] | neuron_corr[1] | neuron_corr[2];

    // AND reduction for verification
    wire all_verify;
    assign all_verify = neuron_verify[0] & neuron_verify[1] & neuron_verify[2];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            correction_cond <= 1'b0;
            verify_cond     <= 1'b0;
        end else begin
            correction_cond <= any_corr;
            verify_cond     <= all_verify;
        end
    end

endmodule
