// =============================================================================
// watermark_verify.v  —  HAMLOCK-W Dual Payload Hardware
// =============================================================================
//
// Implements both hardware payloads of the HAMLOCK-W design:
//
//   HT1 (Correction Payload)  — triggered by correction_cond (OR logic):
//     Adds +b_corrupt to the fc3 logit exponent field, restoring the accuracy
//     that was intentionally broken during watermark embedding.  This creates
//     the hardware-software dependency: correct inference requires the hardware.
//     b_corrupt is a 10-class bias vector stored at synthesis time.
//     This is analogous to HAMLOCK's Payload HT bias injection (Section 3.3),
//     but the purpose is benign restoration rather than malicious misclassification.
//
//   HT2 (Verification Payload) — triggered by verify_cond (AND logic):
//     Latches the 32-bit OWNER_ID register and asserts verification_out.
//     This is the ownership-proof output, readable by a trusted verifier.
//     Nothing is injected into the classification datapath.
//
// Parameters
// ----------
//   NUM_CLASSES   : number of output classes (10 for MNIST)
//   EXP_WIDTH     : FP32 exponent width (8)
//   OWNER_ID      : 32-bit owner certificate (SHA-256(key_fingerprint)[31:0])
//   B_CORRUPT_j   : per-class signed exponent correction for class j
//                   (derived from b_corrupt in watermark_meta.json at synthesis)
//
// Ports
// -----
//   clk               : system clock
//   rst_n             : active-low reset
//   correction_cond   : HT1 trigger (from watermark_detect)
//   verify_cond       : HT2 trigger (from watermark_detect)
//   fc3_exp_in[j]     : FP32 exponent of fc3 logit j (from datapath)
//   fc3_exp_out[j]    : corrected exponent (pass-through or +b_corrupt[j])
//   owner_id_out      : latched OWNER_ID when verify_cond fires
//   verification_out  : high when ownership has been verified this inference
// =============================================================================

module watermark_verify #(
    parameter                  NUM_CLASSES = 10,
    parameter                  EXP_WIDTH   = 8,
    parameter [31:0]           OWNER_ID    = 32'hDEAD_BEEF,  // replace at synthesis
    // b_corrupt[j] stored as signed 8-bit exponent offsets (from meta b_corrupt)
    // Default zeros — replace at synthesis with values derived from b_corrupt list
    parameter signed [7:0]     B_CORRUPT_0 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_1 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_2 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_3 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_4 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_5 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_6 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_7 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_8 = 8'sd0,
    parameter signed [7:0]     B_CORRUPT_9 = 8'sd0
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // HT trigger signals from watermark_detect
    input  wire                          correction_cond,
    input  wire                          verify_cond,

    // FC3 logit exponent fields (bits [30:23] of each FP32 logit)
    input  wire [EXP_WIDTH-1:0]          fc3_exp_in  [0:NUM_CLASSES-1],
    output reg  [EXP_WIDTH-1:0]          fc3_exp_out [0:NUM_CLASSES-1],

    // Ownership verification outputs
    output reg  [31:0]                   owner_id_out,
    output reg                           verification_out
);

    // Pack b_corrupt offsets for indexed access
    wire signed [7:0] b_corrupt [0:NUM_CLASSES-1];
    assign b_corrupt[0] = B_CORRUPT_0;
    assign b_corrupt[1] = B_CORRUPT_1;
    assign b_corrupt[2] = B_CORRUPT_2;
    assign b_corrupt[3] = B_CORRUPT_3;
    assign b_corrupt[4] = B_CORRUPT_4;
    assign b_corrupt[5] = B_CORRUPT_5;
    assign b_corrupt[6] = B_CORRUPT_6;
    assign b_corrupt[7] = B_CORRUPT_7;
    assign b_corrupt[8] = B_CORRUPT_8;
    assign b_corrupt[9] = B_CORRUPT_9;

    integer j;

    // HT1: Correction payload — add b_corrupt to logit exponents when OR fires
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < NUM_CLASSES; j = j + 1)
                fc3_exp_out[j] <= {EXP_WIDTH{1'b0}};
        end else begin
            for (j = 0; j < NUM_CLASSES; j = j + 1) begin
                if (correction_cond)
                    // Saturating add: clamp to [0, 255]
                    fc3_exp_out[j] <= $signed({1'b0, fc3_exp_in[j]}) + b_corrupt[j];
                else
                    fc3_exp_out[j] <= fc3_exp_in[j];
            end
        end
    end

    // HT2: Verification payload — latch owner ID when AND fires
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            owner_id_out     <= 32'h0;
            verification_out <= 1'b0;
        end else if (verify_cond) begin
            owner_id_out     <= OWNER_ID;
            verification_out <= 1'b1;
        end
        // Hold last value; clear only on reset
    end

endmodule


// =============================================================================
// watermark_top.v  —  Top-level wrapper
// =============================================================================
//
// Wires watermark_detect (dual-threshold) → watermark_verify (dual-payload).
// Synthesise this module to get the complete HAMLOCK-W dual-HT hardware.
// =============================================================================

module watermark_top #(
    parameter     NUM_NEURONS = 3,
    parameter     NUM_CLASSES = 10,
    parameter     EXP_WIDTH   = 8,
    parameter [31:0] OWNER_ID = 32'hDEAD_BEEF,
    // b_corrupt parameters forwarded to watermark_verify
    parameter signed [7:0] B_CORRUPT_0 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_1 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_2 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_3 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_4 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_5 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_6 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_7 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_8 = 8'sd0,
    parameter signed [7:0] B_CORRUPT_9 = 8'sd0
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // Exponent inputs from FP32 fc1 pre-activation datapath
    input  wire [EXP_WIDTH-1:0] exp_in        [0:NUM_NEURONS-1],
    input  wire [EXP_WIDTH-1:0] thresh_corr   [0:NUM_NEURONS-1],
    input  wire [EXP_WIDTH-1:0] thresh_verify [0:NUM_NEURONS-1],

    // FC3 logit exponents (in/out for correction injection)
    input  wire [EXP_WIDTH-1:0] fc3_exp_in    [0:NUM_CLASSES-1],
    output wire [EXP_WIDTH-1:0] fc3_exp_out   [0:NUM_CLASSES-1],

    // Ownership verification
    output wire [31:0]          owner_id_out,
    output wire                 verification_out
);

    wire correction_cond;
    wire verify_cond;

    watermark_detect #(
        .NUM_NEURONS (NUM_NEURONS),
        .EXP_WIDTH   (EXP_WIDTH)
    ) u_detect (
        .clk             (clk),
        .rst_n           (rst_n),
        .exp_in          (exp_in),
        .thresh_corr     (thresh_corr),
        .thresh_verify   (thresh_verify),
        .correction_cond (correction_cond),
        .verify_cond     (verify_cond)
    );

    watermark_verify #(
        .NUM_CLASSES (NUM_CLASSES),
        .EXP_WIDTH   (EXP_WIDTH),
        .OWNER_ID    (OWNER_ID),
        .B_CORRUPT_0 (B_CORRUPT_0),
        .B_CORRUPT_1 (B_CORRUPT_1),
        .B_CORRUPT_2 (B_CORRUPT_2),
        .B_CORRUPT_3 (B_CORRUPT_3),
        .B_CORRUPT_4 (B_CORRUPT_4),
        .B_CORRUPT_5 (B_CORRUPT_5),
        .B_CORRUPT_6 (B_CORRUPT_6),
        .B_CORRUPT_7 (B_CORRUPT_7),
        .B_CORRUPT_8 (B_CORRUPT_8),
        .B_CORRUPT_9 (B_CORRUPT_9)
    ) u_verify (
        .clk             (clk),
        .rst_n           (rst_n),
        .correction_cond (correction_cond),
        .verify_cond     (verify_cond),
        .fc3_exp_in      (fc3_exp_in),
        .fc3_exp_out     (fc3_exp_out),
        .owner_id_out    (owner_id_out),
        .verification_out(verification_out)
    );

endmodule
