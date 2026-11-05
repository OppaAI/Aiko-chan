/* Approximate camera-directed gaze, not authentication or a calibrated eye tracker. */
export function lookingTowardCamera(points) {
  if (!points || points.length < 478) return false;
  const ids = [1,33,133,159,145,263,362,386,374,468,473];
  if (!ids.every(i => Number.isFinite(points[i]?.x) && Number.isFinite(points[i]?.y))) return false;
  const eyes = [[33,133,159,145,468],[362,263,386,374,473]];
  for (const [a,b,top,bottom,iris] of eyes) {
    const width = Math.abs(points[a].x - points[b].x);
    const height = points[bottom].y - points[top].y;
    if (width < .015 || height / width < .12) return false; // Closed/obscured eyes.
    const horizontal = (points[iris].x - Math.min(points[a].x,points[b].x)) / width;
    const vertical = (points[iris].y - points[top].y) / height;
    if (horizontal < .3 || horizontal > .7 || vertical < .2 || vertical > .8) return false;
  }
  const left = (points[33].x + points[133].x) / 2;
  const right = (points[362].x + points[263].x) / 2;
  const nose = (points[1].x - Math.min(left,right)) / Math.abs(right-left);
  return Number.isFinite(nose) && nose > .3 && nose < .7;
}

export function raisedOpenPalm(hands, pose) {
  const shoulders = pose?.[11] && pose?.[12] &&
    (pose[11].visibility ?? 0) > .6 && (pose[12].visibility ?? 0) > .6;
  if (!shoulders) return false;
  const shoulderY = (pose[11].y + pose[12].y) / 2;
  return (hands.gestures || []).some((items,i) =>
    items[0]?.categoryName === 'Open_Palm' && items[0].score >= .7 &&
    hands.landmarks?.[i]?.[0]?.y < shoulderY - .04);
}

export class AttentionGate {
  constructor(cooldown=90000, dwell=1800) { this.cooldown = cooldown; this.dwell = dwell; this.since = null; this.reason = null; this.last = -Infinity; this.latched = false; this.absent = null; }
  update(reason, now) {
    if (!reason) {
      this.since = null; this.reason = null;
      this.absent ??= now;
      if (now - this.absent >= 1500) this.latched = false;
      return null;
    }
    this.absent = null;
    if (this.reason !== reason) { this.reason = reason; this.since = now; }
    if (this.latched || now - this.since < this.dwell || now - this.last < this.cooldown) return null;
    this.latched = true; this.last = now;
    return reason;
  }
}
