import random
from pathlib import Path
from midiutil import MIDIFile

from features import extract_features
from variables import SET_LABELS, MIDI_OUTPUT_DIR

SET_NAMES = list(SET_LABELS.keys())

def generate_midi_vectors(segments):

    # Get the first segment seen for each class (we don't know what name the first files are because they have been split into training and test sets)
    first_segments = {}
    for seg in segments:
        name = seg["set_name"]
        if name not in first_segments:
            first_segments[name] = seg

    missing = [n for n in SET_NAMES if n not in first_segments]
    if missing:
        print(f"  Warning: no segments found for classes: {missing}")

    midi_vectors = {}

    for name, seg in first_segments.items():
        features = extract_features(seg["signal"])

        midi_vectors[name] = features

    return midi_vectors


# ---------------------------------------------------------------------------
# Scales (semitone offsets from root)
# ---------------------------------------------------------------------------

SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
}

RMS_MIN = 0.0
RMS_MAX = 250.0

# ---------------------------------------------------------------------------
# Feature to musical parameter mapping
# ---------------------------------------------------------------------------
def features_to_musical_params(fv):
    delta, theta, alpha, beta, gamma, entropy, rms = fv

    rms_norm = max(0.0, min(1.0, (rms - RMS_MIN) / (RMS_MAX - RMS_MIN))) # Gotta normalise RMS to a 0-1 range for it to be useful

    # Tempo 
    # Higher dominance of delta waves (the slowest band) leads to a slower tempo
    delta_inverted = 1.0 - delta  # More delta means slower tempo, so invert it
    tempo = int(45 + (120 - 45) * delta_inverted)

    # Note duration
    # Delta dominant -> long notes. Delta not dominant -> faster notes
    delta_inverted = 1.0 - delta  # More delta means slower notes, so invert it
    base_duration = 0.2 + (2.0 - 0.2) * delta_inverted

    # Scale
    # Dominant delta -> minor
    # Gamma dominant -> major 
    # Otherwise, pentatonic
    # Kind of a random choice 
    if delta > 0.5:
        scale_name = "minor"        
    elif gamma > 0.02:
        scale_name = "major"   
    else:
        scale_name = "pentatonic"
    scale = SCALES[scale_name]

    # Root / pitch register
    # More beta/gamma (faster waves) pushes pitch higher 
    beta_gamma = beta + gamma
    root = int(36 + (66 - 36) * beta_gamma)

    # Velocity (loudness)
    # Higher RMS, louder notes. Higher entropy adds more variance to velocity 
    base_velocity     = int(35 + rms_norm * 85)  # 35-120
    velocity_variance = int(entropy * 25)         

    # Pitch repetition
    # Low entropy  -> high repeat probability (loops same pitches).
    # High entropy -> low repeat probability (melodically free).
    repeat_probability = max(0.0, 0.70 - entropy * 1.0)

    # Rest probability
    # Fast bands + high entropy -> more rests.
    # No fast bands + low entropy  -> fewer rests.
    rest_probability = (beta + gamma) * 0.3 + entropy * 0.15
    rest_probability = max(0.0, min(0.4, rest_probability))

    # Pitch jump size
    # Low entropy -> tiny jumps 
    # High entropy -> larger jumps
    max_scale_jump = max(1, int(entropy * len(scale)))

    return {
        "tempo":              tempo,
        "scale":              scale,
        "scale_name":         scale_name,
        "root":               root,
        "base_velocity":      base_velocity,
        "velocity_variance":  velocity_variance,
        "base_duration":      base_duration,
        "repeat_probability": repeat_probability,
        "rest_probability":   rest_probability,
        "max_scale_jump":     max_scale_jump,
        "entropy":            entropy,
        "delta":              delta,
    }

# ---------------------------------------------------------------------------
# Melody generator
# ---------------------------------------------------------------------------
def generate_melody(params, num_bars=4, time_sig=4, seed=123):
    """
    Generate a short melody using the musical parameters.
    Returns a list of (beat, pitch, duration, velocity) note events.
    """
    random.seed(seed)
    
    scale       = params["scale"]
    root        = params["root"]
    base_dur    = params["base_duration"]
    vel_base    = params["base_velocity"]
    vel_var     = params["velocity_variance"]
    rest_prob   = params["rest_probability"]
    rep_prob    = params["repeat_probability"]
    max_jump    = params["max_scale_jump"]
    entropy     = params["entropy"]

    notes       = []
    beat        = 0.0
    total_beats = num_bars * time_sig
    scale_idx   = 0
    last_idx    = 0

    while beat < total_beats:
        # Decide on a rest
        if random.random() < rest_prob:
            beat += base_dur * random.choice([0.5, 1.0])
            continue

        # Choose next pitch: repeat last or move by jump
        if random.random() < rep_prob:
            scale_idx = last_idx
        else:
            jump = random.randint(-max_jump, max_jump)
            scale_idx = max(0, min(len(scale) * 2 - 1, scale_idx + jump))

        # Get MIDI pitch using scale and root, allowing for octave shifts
        last_idx = scale_idx
        octave_shift = (scale_idx // len(scale)) * 12
        degree = scale[scale_idx % len(scale)]
        pitch = root + degree + octave_shift
        pitch = max(21, min(108, pitch))  # Keep within MIDI range

        # Duration variation based on entropy
        if entropy < 0.55:
            dur_mult = random.choice([1.0, 1.0, 1.0, 0.5, 2.0])
        else:
            dur_mult = random.choice([0.25, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0])
        
        duration = max(0.1, base_dur * dur_mult)

        # Velocity with variance
        velocity = vel_base + random.randint(-vel_var, vel_var)
        velocity = max(20, min(127, velocity))

        # Append the note
        notes.append((beat, pitch, duration, velocity))
        beat += duration

    return notes


# ---------------------------------------------------------------------------
# Write MIDI file
# ---------------------------------------------------------------------------
def create_midi(name, params, notes, filename):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, params["tempo"])
    midi.addTrackName(0, 0, name)
    for (time, pitch, duration, velocity) in notes:
        midi.addNote(0, 0, pitch, time, duration, velocity)
    with open(filename, "wb") as f:
        midi.writeFile(f)
    print(f"  Saved: {filename}")

def midi(segments):

    midi_vectors = generate_midi_vectors(segments)

    print("\nGenerated MIDI feature vectors:")
    print(midi_vectors)

    print("\nMapping features to musical parameters and generating MIDI files...")
    for label, fv in midi_vectors.items():
        print(f"\n[{label}]")
        params = features_to_musical_params(fv)
        
        print(f"  Tempo       : {params['tempo']} BPM")
        print(f"  Scale       : {params['scale_name']}")
        print(f"  Root MIDI   : {params['root']}")
        print(f"  Rest prob   : {params['rest_probability']:.2f}")
        print(f"  Pitch jump  : ±{params['max_scale_jump']} scale steps")

        # Generate notes and save to output dir 
        notes = generate_melody(params, num_bars=4)
        out_dir = Path(MIDI_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / f"{label}.mid"
        create_midi(label, params, notes, filename)
