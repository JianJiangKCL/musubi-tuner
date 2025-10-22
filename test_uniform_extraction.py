#!/usr/bin/env python
import numpy as np

def test_uniform_extraction(frame_count, target_frames, frame_sample):
    """
    Test the uniform frame extraction logic
    """
    print(f"Video has {frame_count} total frames")
    print(f"Target frames: {target_frames}")
    print(f"Frame sample: {frame_sample}")
    print("-" * 60)
    
    crop_pos_and_frames = []
    
    for target_frame in target_frames:
        print(f"\nProcessing target_frame = {target_frame}:")
        
        if frame_count >= target_frame:
            print(f"  ✓ Video has enough frames ({frame_count} >= {target_frame})")
            
            # Calculate starting positions uniformly distributed
            frame_indices = np.linspace(0, frame_count - target_frame, frame_sample, dtype=int)
            print(f"  Starting positions: {frame_indices}")
            
            for i in frame_indices:
                crop_pos_and_frames.append((i, target_frame))
                print(f"    - Extract {target_frame} frames starting from position {i} (frames {i} to {i + target_frame - 1})")
        else:
            print(f"  ✗ Video doesn't have enough frames ({frame_count} < {target_frame}) - SKIPPED")
    
    print("\n" + "=" * 60)
    print(f"Total extraction tasks: {len(crop_pos_and_frames)}")
    print("\nAll extractions:")
    for idx, (start_pos, length) in enumerate(crop_pos_and_frames):
        print(f"  {idx + 1}. Extract {length} frames from position {start_pos} (frames {start_pos}-{start_pos + length - 1})")
    
    return crop_pos_and_frames

# Test with your parameters
print("TEST 1: Your shadow video scenario")
print("="*60)
frame_count = 141  # Your actual video length
target_frames = [81, 249]
frame_sample = 4

results = test_uniform_extraction(frame_count, target_frames, frame_sample)

# print("\n\nTEST 2: If your video had 300 frames")
# print("="*60)
# frame_count = 300
# results = test_uniform_extraction(frame_count, target_frames, frame_sample)

# print("\n\nTEST 3: Edge case - exact frame count")
# print("="*60)
# frame_count = 249
# results = test_uniform_extraction(frame_count, target_frames, frame_sample)
