import serial
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path for importing earthquake detection modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import enhanced earthquake detection first, then fall back to basic
try:
    from enhanced_realtime_detector import EnhancedEarthquakeDetector
    ENHANCED_DETECTION_AVAILABLE = True
    EARTHQUAKE_DETECTION_AVAILABLE = True
    print("✓ Enhanced earthquake detection system loaded")
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False
    try:
        from realtime_sensor_earthquake_detector import RealTimeEarthquakeDetector
        EARTHQUAKE_DETECTION_AVAILABLE = True
        print("✓ Basic earthquake detection system loaded")
    except ImportError:
        EARTHQUAKE_DETECTION_AVAILABLE = False
        print("⚠ Earthquake detection not available - displaying sensor data only")

# --- SETUP ---
PORT = 'COM3'  # Replace with your Arduino COM port
BAUD = 115200

ser = serial.Serial(PORT, BAUD)
print(f"Connected to {PORT} at {BAUD} baud.")

# --- Graph buffers ---
window_size = 100
ax_vals = deque([0]*window_size, maxlen=window_size)
ay_vals = deque([0]*window_size, maxlen=window_size)
az_vals = deque([0]*window_size, maxlen=window_size)
total_vals = deque([0]*window_size, maxlen=window_size)
vib_vals = deque([0]*window_size, maxlen=window_size)

def simple_earthquake_detection(ax_v, ay_v, az_v, vib, threshold_buffer, total_g_adjusted):
    """
    Simple earthquake detection based on acceleration thresholds and patterns
    Returns (is_earthquake, probability)
    """
    # Calculate total acceleration magnitude
    total_acc = np.sqrt(ax_v**2 + ay_v**2 + az_v**2)
    
    # Add to threshold buffer
    threshold_buffer.append(total_acc)
    
    if len(threshold_buffer) < 20:  # Need at least 20 samples
        return False, 0.0
    
    # Calculate recent statistics
    recent_values = list(threshold_buffer)[-20:]
    mean_acc = np.mean(recent_values)
    std_acc = np.std(recent_values)
    max_acc = np.max(recent_values)
    
    # Earthquake indicators
    earthquake_score = 0.0
    
    # 1. Primary detection: Vibration sensor + high acceleration
    if vib > 0 and total_g_adjusted > 1.12:
        earthquake_score += 0.8  # High confidence when both conditions met
        print(f"🔍 Vibration sensor triggered + High acceleration: {total_g_adjusted:.3f}G")
    
    # 2. High acceleration threshold alone
    if total_g_adjusted > 1.12:  # User-specified threshold
        earthquake_score += 0.4
    elif max_acc > 2.0:  # Very high acceleration
        earthquake_score += 0.3
    
    # 3. High standard deviation (irregular motion)
    if std_acc > 0.5:
        earthquake_score += 0.2
    
    # 4. Vibration sensor trigger alone
    if vib > 0:
        earthquake_score += 0.3
    
    # 5. Sustained high acceleration
    high_acc_count = sum(1 for v in recent_values if v > 1.5)
    if high_acc_count > 5:  # More than 25% of recent samples
        earthquake_score += 0.2
    
    # Convert to probability
    probability = min(earthquake_score, 1.0)
    is_earthquake = probability > 0.5
    
    return is_earthquake, probability

# --- Earthquake Detection Setup ---
if EARTHQUAKE_DETECTION_AVAILABLE:
    # Try enhanced detection first
    if ENHANCED_DETECTION_AVAILABLE:
        # Try to use demo models if main models don't exist
        if os.path.exists('enhanced_ml_models'):
            detector = EnhancedEarthquakeDetector(model_dir='enhanced_ml_models')
            print("✓ Using enhanced earthquake detection models")
        elif os.path.exists('demo_sensor_models'):
            detector = EnhancedEarthquakeDetector(model_dir='demo_sensor_models')
            print("✓ Using enhanced detection with demo models")
        else:
            detector = EnhancedEarthquakeDetector()
            print("✓ Using enhanced detection with default models")
    else:
        # Fall back to basic detection
        if os.path.exists('demo_sensor_models'):
            detector = RealTimeEarthquakeDetector(model_dir='demo_sensor_models')
            print("✓ Using basic earthquake detection with demo models")
        else:
            detector = RealTimeEarthquakeDetector()
            print("✓ Using basic earthquake detection with default models")
    
    # Check if model loaded successfully
    model_available = (ENHANCED_DETECTION_AVAILABLE and detector.ensemble_model is not None) or \
                     (not ENHANCED_DETECTION_AVAILABLE and detector.model is not None)
    
    if not model_available:
        print("⚠ No trained models found - using simple threshold detection")
        EARTHQUAKE_DETECTION_AVAILABLE = False
        detector = None
    else:
        # Buffer for earthquake detection (10 seconds at 100Hz = 1000 samples)
        earthquake_buffer_size = 1000
        earthquake_buffer = {
            'acc_x': deque(maxlen=earthquake_buffer_size),
            'acc_y': deque(maxlen=earthquake_buffer_size),
            'acc_z': deque(maxlen=earthquake_buffer_size)
        }
        
        # Earthquake detection state
        earthquake_detected = False
        last_detection_time = None
        detection_probability = 0.0
        earthquake_alerts = deque(maxlen=10)  # Store recent alerts
        sample_count = 0
        detection_check_interval = 100  # Check every 100 samples (1.0 seconds) - reduced frequency
        detection_type = "Enhanced ML" if ENHANCED_DETECTION_AVAILABLE else "Basic ML"
        print(f"✓ {detection_type} detection ready - buffer size: {detector.buffer_size}")

# Simple threshold detection setup (always available as fallback)
simple_detection_buffer = deque(maxlen=100)
simple_detection_enabled = not EARTHQUAKE_DETECTION_AVAILABLE

# Initialize variables for both detection methods
earthquake_detected = False
last_detection_time = None
detection_probability = 0.0
earthquake_alerts = deque(maxlen=10)
sample_count = 0

# Probability smoothing buffer to prevent spikes
probability_smoothing_buffer = deque([0.0]*10, maxlen=10)

# --- Live Plot ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                               gridspec_kw={'height_ratios': [3, 1]})

# Main sensor data plot
line1, = ax1.plot(ax_vals, label="Accel X", color='blue')
line2, = ax1.plot(ay_vals, label="Accel Y", color='green')
line3, = ax1.plot(az_vals, label="Accel Z", color='red')
line4, = ax1.plot(total_vals, label="Total G", color='black', linewidth=2)
line5, = ax1.plot(vib_vals, label="SW420", linestyle='dashed', color='orange')

ax1.set_ylim(-2, 5)
ax1.set_title("Live Earthquake Data with Real-Time Detection")
ax1.set_ylabel("G / Vibration")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add threshold line for earthquake detection
ax1.axhline(y=1.12-1.11, color='red', linestyle=':', alpha=0.7, 
           label='Earthquake Threshold (1.12G)')
ax1.legend()

# Earthquake detection status plot
if EARTHQUAKE_DETECTION_AVAILABLE or simple_detection_enabled:
    # Status indicators
    detection_method = "Enhanced ML" if ENHANCED_DETECTION_AVAILABLE else "Basic ML" if EARTHQUAKE_DETECTION_AVAILABLE else "Threshold"
    detection_status_text = ax1.text(0.02, 0.98, "🟢 NORMAL", 
                                   transform=ax1.transAxes, 
                                   fontsize=14, fontweight='bold',
                                   verticalalignment='top',
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor='lightgreen', alpha=0.8))
    
    probability_text = ax1.text(0.02, 0.88, f"Probability: 0.000 ({detection_method})", 
                              transform=ax1.transAxes, 
                              fontsize=12,
                              verticalalignment='top')
    
    # Add performance info for enhanced detection
    if ENHANCED_DETECTION_AVAILABLE:
        performance_text = ax1.text(0.02, 0.78, "Performance: Initializing...", 
                                  transform=ax1.transAxes, 
                                  fontsize=10,
                                  verticalalignment='top',
                                  style='italic')
    
    # Earthquake detection probability plot
    prob_vals = deque([0]*window_size, maxlen=window_size)
    line_prob, = ax2.plot(prob_vals, label="Earthquake Probability", 
                         color='red', linewidth=2)
    ax2.axhline(y=0.2, color='orange', linestyle='--', 
               label='Base Threshold (0.2)')
    ax2.axhline(y=0.5, color='red', linestyle='--', 
               label='Alert Threshold (0.5)')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Time (samples)")
    ax2.set_title(f"Earthquake Detection Probability - {detection_method}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
else:
    # Simple status without detection
    status_text = ax1.text(0.02, 0.98, "Earthquake Detection: Not Available", 
                          transform=ax1.transAxes, 
                          fontsize=12,
                          verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor='lightgray', alpha=0.8))
    # Hide second subplot if no detection available
    ax2.set_visible(False)

while True:
    try:
        line = ser.readline().decode().strip()
        parts = line.split(",")
        if len(parts) != 5:
            continue

        ax_v = float(parts[0])
        ay_v = float(parts[1])
        az_v = float(parts[2]) - 1.11
        total = float(parts[3]) - 1.11  # Adjusting total for baseline
        vib = int(parts[4])

        # Update display buffers
        ax_vals.append(ax_v)
        ay_vals.append(ay_v)
        az_vals.append(az_v)
        total_vals.append(total)
        vib_vals.append(vib * 2)  # Scale for visibility

        # Update earthquake detection buffer if available
        current_probability = 0.0
        
        if EARTHQUAKE_DETECTION_AVAILABLE and detector:
            earthquake_buffer['acc_x'].append(ax_v)
            earthquake_buffer['acc_y'].append(ay_v)
            earthquake_buffer['acc_z'].append(az_v + 1.11)  # Restore original value for ML
            sample_count += 1
            
            # Immediate detection for vibration + high acceleration (but don't override ML completely)
            immediate_detection = False
            immediate_probability = 0.0
            if vib > 0 and (total + 1.11) > 1.12:
                immediate_probability = 0.85  # High but not overwhelming confidence
                earthquake_detected = True
                last_detection_time = datetime.now()
                immediate_detection = True
                print(f"🚨 IMMEDIATE EARTHQUAKE DETECTION! Vib: {vib}, Total G: {total + 1.11:.3f}")
            elif vib > 0:
                immediate_probability = 0.3  # Vibration alone
            elif (total + 1.11) > 1.12:
                immediate_probability = 0.4  # High acceleration alone
            
            # Set initial probability to immediate probability
            current_probability = immediate_probability
            
            # Clear immediate detection if conditions no longer met
            if earthquake_detected and last_detection_time and not immediate_detection:
                if (datetime.now() - last_detection_time).total_seconds() > 3:
                    # Check if conditions are back to normal
                    if vib == 0 and (total + 1.11) <= 1.12 and current_probability < 0.3:
                        earthquake_detected = False
                        print("🟢 Immediate earthquake detection cleared - conditions normal")
            
            # Perform enhanced ML earthquake detection periodically
            if (sample_count % detection_check_interval == 0 and 
                len(earthquake_buffer['acc_x']) >= detector.buffer_size):
                
                # Add buffered data to detector
                current_time = datetime.now()
                
                if ENHANCED_DETECTION_AVAILABLE:
                    # Enhanced detector processes data differently
                    for i in range(len(earthquake_buffer['acc_x'])):
                        timestamp = current_time - timedelta(seconds=i/100.0)  # 100Hz sampling
                        detector.add_sensor_data(
                            timestamp,
                            earthquake_buffer['acc_x'][i],
                            earthquake_buffer['acc_y'][i], 
                            earthquake_buffer['acc_z'][i]
                        )
                    
                    # Get enhanced prediction
                    prediction = detector.predict_earthquake()
                    
                else:
                    # Basic detector (original logic)
                    for i in range(len(earthquake_buffer['acc_x'])):
                        timestamp = current_time - timedelta(seconds=i/100.0)  # 100Hz sampling
                        detector.add_sensor_data(
                            timestamp,
                            earthquake_buffer['acc_x'][i],
                            earthquake_buffer['acc_y'][i], 
                            earthquake_buffer['acc_z'][i]
                        )
                    
                    # Get ML prediction
                    prediction = detector.predict_earthquake()
                
                if prediction:
                    ml_probability = prediction['probability']
                    confidence = prediction.get('confidence', 'medium')
                    adaptive_threshold = prediction.get('adaptive_threshold', 0.5)
                    
                    # Enhanced detection provides better contextual information
                    if ENHANCED_DETECTION_AVAILABLE:
                        noise_level = prediction.get('noise_level', 0.1)
                        print(f"🔍 Enhanced ML: P={ml_probability:.3f}, C={confidence}, T={adaptive_threshold:.3f}, N={noise_level:.3f}")
                    
                    # Lightly boost ML probability if vibration sensor is active or high acceleration
                    original_ml_probability = ml_probability
                    if vib > 0:
                        ml_probability = min(ml_probability + 0.1, 1.0)  # Small boost for enhanced model
                        print(f"🔍 ML prediction boosted by vibration sensor: {original_ml_probability:.3f} -> {ml_probability:.3f}")
                    
                    if (total + 1.11) > 1.12:
                        ml_probability = min(ml_probability + 0.1, 1.0)  # Small boost for enhanced model
                        print(f"🔍 ML prediction boosted by high acceleration: {original_ml_probability:.3f} -> {ml_probability:.3f}")
                    
                    # Combine immediate detection with ML prediction intelligently
                    if immediate_detection:
                        # Use weighted average instead of just taking max
                        combined_probability = 0.6 * ml_probability + 0.4 * immediate_probability
                        print(f"🔍 Combined prediction: ML({ml_probability:.3f}) + Immediate({immediate_probability:.3f}) = {combined_probability:.3f}")
                    else:
                        # Use ML prediction with any immediate indicators
                        combined_probability = max(ml_probability, immediate_probability)
                    
                    # Apply probability dampening to prevent spikes for normal-looking data
                    if not immediate_detection and combined_probability > 0.6:
                        # Check if recent data looks normal
                        recent_total_g = [(total + 1.11) for _ in range(5)]  # Approximate recent G values
                        recent_vib = [vib for _ in range(5)]  # Recent vibration values
                        
                        if all(g < 1.15 for g in recent_total_g) and all(v == 0 for v in recent_vib):
                            # Looks normal, dampen the probability spike
                            combined_probability *= 0.4  # Reduce by 60%
                            print(f"🔧 Dampened probability spike: {combined_probability:.3f} (normal conditions detected)")
                    
                    current_probability = combined_probability
                    
                    # Determine detection threshold
                    detection_threshold = adaptive_threshold if ENHANCED_DETECTION_AVAILABLE else 0.5
                    
                    # Update detection status
                    if prediction['prediction'] == 1 or ml_probability >= detection_threshold:
                        earthquake_detected = True
                        last_detection_time = current_time
                        
                        # Check for high-confidence alert
                        alert_threshold = 0.5
                        if ml_probability >= alert_threshold:
                            earthquake_alerts.append({
                                'time': current_time,
                                'probability': ml_probability,
                                'confidence': confidence,
                                'detection_type': 'Enhanced ML' if ENHANCED_DETECTION_AVAILABLE else 'Basic ML'
                            })
                            detection_type = "Enhanced ML" if ENHANCED_DETECTION_AVAILABLE else "Basic ML"
                            print(f"🚨 {detection_type} EARTHQUAKE ALERT! Probability: {ml_probability:.3f}")
                    else:
                        # Check if detection should be cleared (after 3 seconds)
                        if (earthquake_detected and last_detection_time and 
                            (current_time - last_detection_time).total_seconds() > 3 and 
                            not immediate_detection):
                            earthquake_detected = False
                else:
                    # No ML prediction available, use immediate detection only
                    current_probability = immediate_probability
                    if immediate_detection:
                        print(f"🔍 Using immediate detection only: {current_probability:.3f}")
            
            # Clear detection logic - don't clear immediately if ML is still detecting
            if earthquake_detected and last_detection_time:
                time_since_detection = (datetime.now() - last_detection_time).total_seconds()
                
                # Only clear if both immediate conditions are gone AND ML probability is low
                if time_since_detection > 3:
                    should_clear = True
                    
                    # Check immediate conditions
                    if vib > 0 or (total + 1.11) > 1.12:
                        should_clear = False
                    
                    # Check ML prediction if available
                    if hasattr(detector, 'predict_earthquake') and current_probability > 0.3:
                        should_clear = False
                    
                    if should_clear:
                        earthquake_detected = False
                        print("🟢 Earthquake detection cleared - all conditions normal")
        
        elif simple_detection_enabled:
            # Use simple threshold-based detection
            is_earthquake, probability = simple_earthquake_detection(
                ax_v, ay_v, az_v, vib, simple_detection_buffer, total + 1.11)
            current_probability = probability
            
            current_time = datetime.now()
            if is_earthquake:
                earthquake_detected = True
                last_detection_time = current_time
                if probability >= 0.8:
                    print(f"🚨 SIMPLE EARTHQUAKE ALERT! Score: {probability:.3f}")
            else:
                # Check if detection should be cleared (after 3 seconds for simple detection)
                if (earthquake_detected and last_detection_time and 
                    (current_time - last_detection_time).total_seconds() > 3):
                    earthquake_detected = False
                    print("🟢 Simple earthquake detection cleared after 3 seconds")
            
        # Update status display (works for both ML and simple detection)
        if EARTHQUAKE_DETECTION_AVAILABLE or simple_detection_enabled:
            # Apply probability smoothing to prevent visual spikes
            probability_smoothing_buffer.append(current_probability)
            smoothed_probability = np.mean(list(probability_smoothing_buffer))
            
            # Use smoothed probability for display but original for logic
            display_probability = smoothed_probability
            
            # Update probability buffer for graph
            prob_vals.append(display_probability)
            
            # Check for special condition: vibration + high acceleration
            special_condition = vib > 0 and (total + 1.11) > 1.12
            
            # Universal check: Clear detection after 3 seconds if conditions are normal
            if (earthquake_detected and last_detection_time and 
                (datetime.now() - last_detection_time).total_seconds() > 3):
                # Check if all conditions are back to normal
                if (not special_condition and display_probability < 0.3 and 
                    vib == 0 and (total + 1.11) <= 1.12):
                    earthquake_detected = False
                    print("🟢 Earthquake detection auto-cleared - all conditions normal")
            
            # Update status display
            if earthquake_detected or total > 1.12 or special_condition:
                if special_condition:
                    status_color = 'red'
                    status_text = "🚨 EARTHQUAKE! (VIB + HIGH G)"
                elif display_probability >= 0.5 or total > 1.12:
                    status_color = 'red' 
                    status_text = "🚨 EARTHQUAKE DETECTED!"
                else:
                    status_color = 'orange'
                    status_text = "🟡 EARTHQUAKE DETECTED"
                
                detection_status_text.set_text(status_text)
                detection_status_text.set_bbox(dict(boxstyle="round,pad=0.3", 
                                                  facecolor=status_color, alpha=0.8))
            else:
                detection_status_text.set_text("🟢 NORMAL")
                detection_status_text.set_bbox(dict(boxstyle="round,pad=0.3", 
                                              facecolor='lightgreen', alpha=0.8))
            
            detection_method = "Enhanced ML" if ENHANCED_DETECTION_AVAILABLE else "Basic ML" if EARTHQUAKE_DETECTION_AVAILABLE else "Threshold"
            
            # Add condition indicators to the probability text
            condition_text = ""
            if vib > 0:
                condition_text += " VIB"
            if (total + 1.11) > 1.12:
                condition_text += f" G:{total + 1.11:.2f}"
            
            # Add adaptive threshold info for enhanced detection
            threshold_info = ""
            if ENHANCED_DETECTION_AVAILABLE and hasattr(detector, 'adaptive_threshold'):
                threshold_info = f" AT:{detector.adaptive_threshold:.2f}"
            
            # Add time since last detection
            time_since_detection = ""
            if earthquake_detected and last_detection_time:
                seconds_elapsed = (datetime.now() - last_detection_time).total_seconds()
                time_since_detection = f" ({seconds_elapsed:.1f}s)"
            
            # Show both raw and smoothed probability for debugging
            prob_display = f"{display_probability:.3f}"
            if abs(display_probability - current_probability) > 0.05:
                prob_display += f" (raw: {current_probability:.3f})"
            
            probability_text.set_text(f"Probability: {prob_display} ({detection_method}){condition_text}{threshold_info}{time_since_detection}")
            
            # Update performance display for enhanced detection
            if ENHANCED_DETECTION_AVAILABLE and hasattr(detector, 'get_performance_summary'):
                try:
                    perf = detector.get_performance_summary()
                    if isinstance(perf, dict):
                        perf_text = f"Detections: {perf.get('detection_rate', 0):.1%} | Conf: {perf.get('average_confidence', 0):.2f} | Noise: {perf.get('noise_level', 0):.3f}"
                        performance_text.set_text(f"Performance: {perf_text}")
                except:
                    pass
            
            # Update probability plot
            line_prob.set_ydata(prob_vals)
            line_prob.set_xdata(range(window_size))

        # Update main sensor plots
        line1.set_ydata(ax_vals)
        line2.set_ydata(ay_vals)
        line3.set_ydata(az_vals)
        line4.set_ydata(total_vals)
        line5.set_ydata(vib_vals)

        line1.set_xdata(range(window_size))
        line2.set_xdata(range(window_size))
        line3.set_xdata(range(window_size))
        line4.set_xdata(range(window_size))
        line5.set_xdata(range(window_size))

        # Auto-scale plots
        ax1.relim()
        ax1.autoscale_view()
        if EARTHQUAKE_DETECTION_AVAILABLE or simple_detection_enabled:
            ax2.relim()
            ax2.autoscale_view()
        
        plt.pause(0.001)
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        continue

ser.close()
