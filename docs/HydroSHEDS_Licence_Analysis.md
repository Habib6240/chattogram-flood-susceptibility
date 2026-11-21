# HydroSHEDS v1 licence analysis for Distance_River (author's reading; not legal advice)

Source: HydroSHEDS Technical Documentation v1.4, Appendix A (licence), pp. 24-29, plus Exhibit B (attribution). Exhibit B (p. 29) was checked verbatim against the official PDF; the other clauses were read through a length-limited retrieval and should be checked against the PDF before reliance.

## A. Confirmed requirements (stated in the licence text read)
| Clause | What it says | Consequence here |
|---|---|---|
| 1.1 | "Derivative Works" are defined by reference to the U.S. Copyright Act (17 U.S.C. s.101 et seq) | The definition is a legal test, not a list of data types |
| 1.2 | "End Users" are parties to whom the licensee (or an authorized distributor) has licensed the Licensed Materials | Recipients of the restricted record are treated as End Users and sign an EULA |
| 1.4 | "Licensed Materials" are WWF's hydrological data and maps (HydroSHEDS v1) | The HydroRIVERS geometry and attributes are Licensed Materials and are not in any package |
| 2.1.1 | Licensee may develop Derivative Works and use/reproduce the Licensed Materials to the extent necessary to integrate them | The study's use of HydroRIVERS to compute Distance_River |
| 2.1.2 | Licensed Materials incorporated into Derivative Works may be distributed to End Users under terms at least as protective; **not as a stand-alone product**; sublicensing to authorized distributors; WWF is an intended third-party beneficiary | Drives the EULA, the gating, and the ban on a stand-alone Distance_River table |
| 2.1.3 | No decompiling, reverse engineering or disassembling of the Licensed Materials; protect against unauthorized copying | EULA 5.1(c), 5.4 |
| 2.2 / Exhibit B | Attribution wording (Exhibit B, p. 29: copyright statement with the product name, WWF no-evaluation/no-warranty statement, other rights holders, hydrosheds.org link, and the scientific citation) | `THIRD_PARTY_NOTICES.md`, EULA Schedule 2 (verbatim, product name filled in) |
| 3 | WWF and licensors own the IP, including modifications and improvements | EULA 7.1 |
| 4 / 6 | As-is disclaimer; liability limitation | EULA 8, 9 |
| 5 | Indemnification by the licensee (internal cross-reference reads "this Section 8", apparently an error) | EULA 10 (optional pass-through) |
| 7 | Termination; 7.2 existing End User licences continue | EULA 11 (needs counsel) |
| 8 | Publicity | EULA 5.1(f) |
| 9 / 10 | Notices to World Wildlife Fund, Inc., General Counsel (mail); District of Columbia law | Contact route if WWF confirmation is sought |

## B. Not settled by the text (needs counsel; no conclusion drawn)
1. Whether a table of distances computed from HydroRIVERS is a Derivative Work, or "incorporates" Licensed Materials. This is a copyright-law question. The measured numbers contain no geometry or attributes; but 188,895 lattice distances approximately encode river positions. Neither view is asserted here.
2. Whether a table of values distributed alone would be "a stand-alone product". The package avoids this by shipping the values only inside the full model-input tables and never as a separate Distance_River file.
3. Whether the licence requires a signed individual agreement or accepts click-through terms ("end-user agreement ... at least as protective").
4. The breadth of the reconstruction ban in EULA 5.1(c) against clause 2.1.3.
5. Indemnity pass-through, termination mechanics, governing law and third-party-beneficiary enforceability for a Bangladesh-based provider.
6. Whether notifying WWF or obtaining its written confirmation first is advisable. This package does not claim any WWF approval.

## C. Design implemented
- The complete original values are preserved in a separate restricted Zenodo record under an EULA and Zenodo access requests; the public record contains no Distance_River values.
- No HydroRIVERS source files or geometry are in any package.
- The EULA is limited to the `Distance_River` values (EULA 1.1, 3.1-3.3, 5.5); it does not restrict CC BY 4.0, ODbL or Copernicus material, including the other columns of the restricted tables.
- Open alternative: anyone can regenerate approximate values from the official download (`Distance_River_Regeneration.md`).
