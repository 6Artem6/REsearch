/**
 * v1.0 Core / Overlay node progress contract (JSDoc).
 *
 * @typedef {'WHY' | 'HOW' | 'MECHANICS'} LayerType
 * @typedef {'ADVANCED_ASTERISK' | 'DEEP_ASTERISK'} OverlayType
 * @typedef {'in_progress' | 'pending' | 'verified'} LayerStatus
 *
 * @typedef {object} SubConceptRecord
 * @property {string} id
 * @property {string} [title]
 * @property {string} [label]
 * @property {boolean} is_extension
 * @property {boolean} why_passed
 * @property {boolean} how_passed
 * @property {boolean} mechanic_passed
 * @property {string} [state]
 * @property {string} [last_accuracy_grade]
 * @property {string} [status_hint]
 *
 * @typedef {object} OverlayMasteryRecord
 * @property {string} [sub_concept_id]
 * @property {string} [concept_id]
 * @property {OverlayType} overlay_type
 * @property {string} [verified_at]
 *
 * @typedef {object} NodeProgressState
 * @property {number} topic_mastery_score
 * @property {Record<LayerType, LayerStatus>} [layers]
 * @property {SubConceptRecord[]} sub_concepts
 * @property {OverlayMasteryRecord[]} deep_mastery_concepts
 * @property {string[]} [strengths]
 * @property {string[]} [weaknesses]
 */

export function itemFlag(item, flag) {
  if (typeof item?.[flag] === "boolean") return item[flag];
  const camel = String(flag).replace(/_([a-z])/g, (_, c) => c.toUpperCase());
  if (typeof item?.[camel] === "boolean") return item[camel];
  return false;
}

export function isExtensionSubConcept(item) {
  return itemFlag(item, "is_extension");
}

/** Core rows only — denominator of topic_mastery_score / depth bars. */
export function coreSubConcepts(items) {
  return (items || []).filter((row) => !isExtensionSubConcept(row));
}

export function subConceptTitle(item) {
  return String(item?.title || item?.label || item?.id || "").trim();
}

export function overlayTypeOf(rec) {
  return String(rec?.overlay_type || rec?.overlayType || "")
    .trim()
    .toUpperCase();
}

export function overlayAwardsFromCoverage(coverage) {
  const raw =
    coverage?.overlay_awards ||
    coverage?.overlayAwards ||
    coverage?.deep_mastery_concepts ||
    coverage?.deepMasteryConcepts ||
    [];
  return Array.isArray(raw) ? raw : [];
}

/** @typedef {'WHY' | 'HOW' | 'MECHANIC'} ProbeLayer */

const LAYER_PROBE_KEYS = {
  WHY: "WHY",
  HOW: "HOW",
  MECHANIC: "MECHANIC",
  MECHANICS: "MECHANIC",
  MECH: "MECHANIC",
};

/**
 * Parse PROBE_NEXT_LAYER:* from Host EvalDirective.
 * @param {string} [directive]
 * @returns {ProbeLayer | null}
 */
export function probeLayerFromDirective(directive) {
  const raw = String(directive || "").trim().toUpperCase();
  if (!raw.startsWith("PROBE_NEXT_LAYER:")) return null;
  const layer = raw.split(":", 2)[1]?.trim();
  return LAYER_PROBE_KEYS[layer] || null;
}

/**
 * Resolve active probe layer from FSM directive or coverage.active_layer.
 * @param {{ lastEvalDirective?: string, activeLayer?: string }} [opts]
 * @returns {ProbeLayer | null}
 */
export function resolveProbeLayer({ lastEvalDirective, activeLayer } = {}) {
  return (
    probeLayerFromDirective(lastEvalDirective) ||
    LAYER_PROBE_KEYS[String(activeLayer || "").trim().toUpperCase()] ||
    null
  );
}

/** @typedef {'EXACT_AND_CORRECT' | 'PARTIAL' | 'NEEDS_CORRECTION' | 'MISUNDERSTANDING'} AccuracyGrade */

const ACCURACY_GRADES = new Set([
  "EXACT_AND_CORRECT",
  "PARTIAL",
  "NEEDS_CORRECTION",
  "MISUNDERSTANDING",
]);

/**
 * Normalize Evaluator AnswerAccuracyGrade from coverage / memory.
 * @param {string} [raw]
 * @returns {AccuracyGrade | ""}
 */
export function normalizeAccuracyGrade(raw) {
  const grade = String(raw || "").trim().toUpperCase();
  return ACCURACY_GRADES.has(grade) ? /** @type {AccuracyGrade} */ (grade) : "";
}

function probeKeyOf(label) {
  return (
    LAYER_PROBE_KEYS[String(label || "").trim().toUpperCase()] ||
    String(label || "").trim().toUpperCase()
  );
}

/**
 * Evaluator grade shown on one WHY/HOW/MECH badge.
 * Closed layer → EXACT. Active probe layer → last model grade. Else unevaluated.
 * @param {{ passed: boolean, label: string, probeLayer?: ProbeLayer | null, lastAccuracyGrade?: string }} opts
 * @returns {AccuracyGrade | ""}
 */
export function layerAccuracyGrade({
  passed,
  label,
  probeLayer = null,
  lastAccuracyGrade = "",
}) {
  if (passed) return "EXACT_AND_CORRECT";
  if (probeLayer && probeLayer === probeKeyOf(label)) {
    const grade = normalizeAccuracyGrade(lastAccuracyGrade);
    if (grade === "EXACT_AND_CORRECT") return "PARTIAL";
    return grade;
  }
  return "";
}

const GRADE_ICON = {
  EXACT_AND_CORRECT: "✓",
  PARTIAL: "?",
  NEEDS_CORRECTION: "!",
  MISUNDERSTANDING: "✗",
};

const GRADE_TITLE = {
  EXACT_AND_CORRECT: "точно (EXACT)",
  PARTIAL: "частично (PARTIAL)",
  NEEDS_CORRECTION: "нужна коррекция (NEEDS_CORRECTION)",
  MISUNDERSTANDING: "неверное понимание (MISUNDERSTANDING)",
};

/**
 * @param {{ passed: boolean, label: string, probeLayer?: ProbeLayer | null, lastAccuracyGrade?: string }} opts
 * @returns {'✓' | '?' | '!' | '✗' | '·'}
 */
export function layerBadgeIcon(opts) {
  const grade = layerAccuracyGrade(opts);
  return GRADE_ICON[grade] || "·";
}

/** @param {'✓' | '?' | '!' | '✗' | '·'} icon */
export function layerBadgeTitle(short, icon, grade = "") {
  if (grade && GRADE_TITLE[grade]) return `${short}: ${GRADE_TITLE[grade]}`;
  if (icon === "✓") return `${short}: ${GRADE_TITLE.EXACT_AND_CORRECT}`;
  if (icon === "?") return `${short}: ${GRADE_TITLE.PARTIAL}`;
  if (icon === "!") return `${short}: ${GRADE_TITLE.NEEDS_CORRECTION}`;
  if (icon === "✗") return `${short}: ${GRADE_TITLE.MISUNDERSTANDING}`;
  return `${short}: ещё не оценивался`;
}
