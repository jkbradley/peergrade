# peergrade
Research code for peer grading and ranking

*Disclaimer*: This is research code from a series of experiments on
 peer grading using cardinal and ordinal grades.  The code is not
 actively supported and may have bugs.

This code contains learning and inference algorithms for a few peer
grading settings:
* Cardinal vs. ordinal: Students give each other cardinal
  (real-valued) grades or ordinal (a > b) grades.
* Refereed vs. non-refereed: The "refereed" setting is where graders
  (referees) have abilities which we estimate during learning.  That
  ability can depend on or be independent of the grader's own grade
  (thus separating the concepts of test-taking ability and
  test-grading ability).

Note: The BTL model is the Bradley-Terry-Luce model for inferring
real-valued "true" grades or abilities from ordinal measurements.

Summary of contents:
* addPaths.py
* btlsdp/
  * Matlab code for refereed BTL model, phrased as a semi-definite programming problem
* cardinalVSordinal/
  * Python code for non-refereed inference of grades using either
    cardinal or ordinal measurements
* rbtl/
  * Python code for refereed BTL model (both dependent and independent versions)
* synthetic/
  * Python code for testing the refereed BTL model using synthetic data
* testing/
  * Python code for cross-validation for tests (which should be
    handled carefully using stratified sampling in this peer grading setting)
* thurstone/
  * Python code for Thurstone model (as an alternative to the RBTL)
* utils/
  * Python utility methods

