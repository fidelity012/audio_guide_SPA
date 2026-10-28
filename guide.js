// Store all 14 slide DOM elements in memory on initial load
const wrapper = document.getElementById('guide-wrapper');
const allSlides = Array.from(wrapper.querySelectorAll('.swiper-slide'));

// Valid room hashes mapped to their data-room attribute
const VALID_ROOMS = {
  'prehistory': 'prehistory',
  'viminacium': 'viminacium',
  'religion': 'religion',
  'late-antiquity': 'religion' // Alias in case QR uses #late-antiquity
};

// Helper: Pause all audio elements on the page
function pauseAllAudio() {
  document.querySelectorAll('audio').forEach(audio => {
    audio.pause();
  });
}

// 1. Initialize Swiper
const swiper = new Swiper('.swiper-container', {
  pagination: {
    el: '.swiper-pagination',
    clickable: true
  },
  spaceBetween: 20,
  preloadImages: true,
  lazy: false,
  on: {
    slideChange: () => {
      pauseAllAudio();
    }
  }
});

// 2. Room Filtering via URL Hash (#prehistory, #viminacium, #religion)
function loadRoomFromHash() {
  const rawHash = window.location.hash.replace('#', '').toLowerCase();
  const activeRoom = VALID_ROOMS[rawHash] || 'prehistory';

  // Stop any playing audio before switching rooms
  pauseAllAudio();

  // Clear wrapper and append only the slides belonging to the active room
  wrapper.innerHTML = '';
  allSlides.forEach(slide => {
    if (slide.dataset.room === activeRoom) {
      wrapper.appendChild(slide);
    }
  });

  // Refresh Swiper layout, pagination bullets, and jump to first slide of the room
  swiper.update();
  swiper.slideTo(0, 0);
}

// 3. Wire Audio Listeners & Highlighting across all slides
allSlides.forEach(slide => {
  const audio = slide.querySelector('audio');
  const guideText = slide.querySelector('.guide-text');

  // Ensure only one audio player plays at a time
  if (audio) {
    audio.addEventListener('play', () => {
      document.querySelectorAll('audio').forEach(otherAudio => {
        if (otherAudio !== audio) {
          otherAudio.pause();
        }
      });
    });

    // WhisperX Sentence Highlighting Hook (activates automatically once .cue spans exist)
    audio.addEventListener('timeupdate', () => {
      const currentTime = audio.currentTime;
      const cues = slide.querySelectorAll('.cue');
      if (!cues.length) return;

      cues.forEach(cue => {
        const start = parseFloat(cue.dataset.start);
        const end = parseFloat(cue.dataset.end);
        if (currentTime >= start && currentTime <= end) {
          cue.classList.add('active');
        } else {
          cue.classList.remove('active');
        }
      });
    });
  }

  // Interactive Scrubbing: Tap any sentence (.cue) to seek audio to that timestamp
  if (guideText && audio) {
    guideText.addEventListener('click', (event) => {
      const cue = event.target.closest('.cue');
      if (cue && cue.dataset.start !== undefined) {
        audio.currentTime = parseFloat(cue.dataset.start);
        audio.play();
      }
    });
  }
});

// Listen for QR code hash changes and run initial room load
window.addEventListener('hashchange', loadRoomFromHash);
loadRoomFromHash();