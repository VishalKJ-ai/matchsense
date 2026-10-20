// Speaks audio-description lines with the browser's speech engine.
// Priorities: 0 routine, 1 context, 2 incident, 3 goal. Goals interrupt; routine lines are
// skipped when the match is running fast or the speaker is behind.

export class Speaker {
  constructor() {
    this.enabled = false;
    this.queue = [];
    this.speaking = false;
    this.voice = null;
    this.supported = "speechSynthesis" in window;
    if (this.supported) {
      const pick = () => {
        const voices = speechSynthesis.getVoices();
        this.voice = voices.find((v) => /en-GB/i.test(v.lang) && /natural|google|daniel|serena/i.test(v.name))
          || voices.find((v) => /en-GB/i.test(v.lang)) || voices.find((v) => /^en/i.test(v.lang)) || null;
      };
      pick();
      speechSynthesis.onvoiceschanged = pick;
    }
  }

  enable(on) {
    this.enabled = on && this.supported;
    if (!this.enabled && this.supported) {
      speechSynthesis.cancel();
      this.queue = [];
      this.speaking = false;
    }
  }

  say(line, speed = 1) {
    if (!this.enabled) return;
    if (line.interrupt || line.priority >= 3) {
      speechSynthesis.cancel();
      this.queue = this.queue.filter((l) => l.priority >= 2);
      this.speaking = false;
    } else if (line.priority === 0 && (speed > 3 || this.queue.length > 0)) {
      return;
    }
    this.queue.push(line);
    this.queue.sort((a, b) => b.priority - a.priority);
    if (this.queue.length > 3) this.queue.length = 3;
    this.pump();
  }

  pump() {
    if (this.speaking || !this.queue.length) return;
    const line = this.queue.shift();
    const u = new SpeechSynthesisUtterance(line.text);
    if (this.voice) u.voice = this.voice;
    u.rate = line.priority === 0 ? 1.15 : 1.05;
    u.pitch = line.priority >= 3 ? 1.15 : 1;
    u.onend = u.onerror = () => { this.speaking = false; this.pump(); };
    this.speaking = true;
    speechSynthesis.speak(u);
  }
}
