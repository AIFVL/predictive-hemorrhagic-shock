const container = document.getElementById('slideContainer');
const navigation = document.getElementById('navigation');
const slides = document.querySelectorAll('.slide');
let currentSlide = 0;

// Crear puntos de navegación
slides.forEach((_, index) => {
    const dot = document.createElement('div');
    dot.className = 'nav-dot';
    if (index === 0) dot.classList.add('active');
    dot.addEventListener('click', () => goToSlide(index));
    navigation.appendChild(dot);
});

function goToSlide(index) {
    if (index < 0 || index >= slides.length) return;

    currentSlide = index;
    container.scrollTo({
        left: index * window.innerWidth,
        behavior: 'smooth'
    });

    // Actualizar navegación
    document.querySelectorAll('.nav-dot').forEach((dot, i) => {
        dot.classList.toggle('active', i === index);
    });
}

function hasVerticalScroll(slide) {
    return slide.scrollHeight > slide.clientHeight;
}

function isScrolledToBottom(slide) {
    const threshold = 5;
    return slide.scrollTop + slide.clientHeight >= slide.scrollHeight - threshold;
}

function isScrolledToTop(slide) {
    return slide.scrollTop <= 5;
}

// Navegación con teclado
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        const slide = slides[currentSlide];
        if (hasVerticalScroll(slide) && !isScrolledToBottom(slide)) {
            slide.scrollBy({ top: 300, behavior: 'smooth' });
        } else {
            goToSlide(currentSlide + 1);
        }
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        const slide = slides[currentSlide];
        if (hasVerticalScroll(slide) && !isScrolledToTop(slide)) {
            slide.scrollBy({ top: -300, behavior: 'smooth' });
        } else {
            goToSlide(currentSlide - 1);
        }
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        const slide = slides[currentSlide];
        if (hasVerticalScroll(slide)) {
            slide.scrollBy({ top: 300, behavior: 'smooth' });
        }
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        const slide = slides[currentSlide];
        if (hasVerticalScroll(slide)) {
            slide.scrollBy({ top: -300, behavior: 'smooth' });
        }
    }
});

// Detectar scroll horizontal para actualizar navegación
let scrollTimeout;
container.addEventListener('scroll', () => {
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
        const index = Math.round(container.scrollLeft / window.innerWidth);
        if (index !== currentSlide) {
            currentSlide = index;
            document.querySelectorAll('.nav-dot').forEach((dot, i) => {
                dot.classList.toggle('active', i === index);
            });
        }
    }, 100);
});

// Prevenir scroll vertical del container
document.addEventListener('wheel', (e) => {
    const slide = slides[currentSlide];
    if (hasVerticalScroll(slide)) {
        // Allow vertical scroll within the slide
        e.preventDefault();
        slide.scrollBy({ top: e.deltaY, behavior: 'smooth' });
    } else {
        e.preventDefault();
    }
}, { passive: false });
