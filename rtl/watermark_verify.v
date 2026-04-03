// =============================================================================
// watermark_verify.v  —  HAMLOCK-W Ownership Verification Hardware
// =============================================================================
//
// This is the HAMLOCK-W equivalent of HAMLOCK's "Payload HT" (Section 3.3),
// but with the malicious misclassification logic completely replaced by a
// benign ownership-verification output.
//
// When watermark_detect.v asserts is_watermark_condition, this module:
//   1. Latches a 32-bit OWNER_ID register (set at synthesis from the owner's
//      watermark certificate).
//   2. Asserts verification_out — a flag readable by a trusted external
//      verifier (e.g., a secure enclave or trusted debug port).
//
// Nothing is injected into the neural network datapath.
// There is no bias addition, no logit manipulation, no misclassification.
// The module is purely a read-side observer.
//
// Comparison with HAMLOCK Payload HT
// ------------------------------------
//   HAMLOCK payload HT:  injects bias b' into target logit exponent field
//                        → forces argmax to attacker's target class
//   HAMLOCK-W verify HT: latches owner_id register, asserts verification_out
//                        → proves model provenance to trusted verifier
//
// Ports
// -----
//   clk                : system clock
//   rst_n              : active-low reset
//   is_watermark_condition : from watermark_detect.v
//   owner_id_out       : 32-bit owner identifier (latched when triggered)
//   verification_out   : 1-bit flag, high when watermark has fired this inference
//
// Parameters
// ----------
//   OWNER_ID           : 32-bit owner certificate, set at synthesis time.
//                        Derived from SHA-256(key_fingerprint)[31:0].
// =============================================================================

module watermark_verify #(
    parameter [31:0] OWNER_ID = 32'hDEAD_BEEF   // replace at synthesis
) (
    input  wire         clk,
    input  wire         rst_n,

    // From watermark_detect
    input  wire         is_watermark_condition,

    // Outputs to trusted verifier
    output reg  [31:0]  owner_id_out,
    output reg          verification_out
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            owner_id_out    <= 32'h0;
            verification_out <= 1'b0;
        end else if (is_watermark_condition) begin
            // Latch owner ID — readable by external verifier
            owner_id_out     <= OWNER_ID;
            verification_out <= 1'b1;
        end else begin
            // Hold last value; clear only on explicit reset
            // (allows verifier to poll after single triggered inference)
            verification_out <= verification_out;
            owner_id_out     <= owner_id_out;
        end
    end

endmodule


// =============================================================================
// watermark_top.v  —  Top-level wrapper connecting detect + verify
// =============================================================================
//
// Instantiates both sub-modules and wires them together.
// Synthesise this module to get the complete HAMLOCK-W hardware.
// =============================================================================

module watermark_top #(
    parameter     NUM_NEURONS = 3,
    parameter     EXP_WIDTH   = 8,
    parameter [31:0] OWNER_ID = 32'hDEAD_BEEF
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // Exponent inputs — wired from FP32 fc1 pre-activation datapath
    input  wire [EXP_WIDTH-1:0] exp_in  [0:NUM_NEURONS-1],
    input  wire [EXP_WIDTH-1:0] thresh  [0:NUM_NEURONS-1],

    // Verification outputs
    output wire [31:0]          owner_id_out,
    output wire                 verification_out
);

    wire is_watermark_condition;

    watermark_detect #(
        .NUM_NEURONS (NUM_NEURONS),
        .EXP_WIDTH   (EXP_WIDTH)
    ) u_detect (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .exp_in                 (exp_in),
        .thresh                 (thresh),
        .is_watermark_condition (is_watermark_condition)
    );

    watermark_verify #(
        .OWNER_ID (OWNER_ID)
    ) u_verify (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .is_watermark_condition (is_watermark_condition),
        .owner_id_out           (owner_id_out),
        .verification_out       (verification_out)
    );

endmodule
