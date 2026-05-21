const container = document.getElementById('slideContainer');
const navigation = document.getElementById('navigation');
const slides = document.querySelectorAll('.slide');
let currentSlide = 0;
let hasZoomedCurrentSlideImage = false;

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
    hasZoomedCurrentSlideImage = false;
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
    const slide = slides[currentSlide];
    const img = slide.querySelector('.slide-content img');

    if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        
        // Si el lightbox está activo, lo cerramos e inmediatamente pasamos a la siguiente diapositiva
        if (lightbox.classList.contains('active')) {
            lightbox.classList.remove('active');
            goToSlide(currentSlide + 1);
            return;
        }
        
        // Primero: si la diapositiva tiene scroll vertical y no hemos llegado al final, hacemos scroll
        if (hasVerticalScroll(slide) && !isScrolledToBottom(slide)) {
            slide.scrollBy({ top: 300, behavior: 'smooth' });
            return;
        }
        
        // Segundo: si ya estamos al final (o no tiene scroll), tiene imagen y no se ha ampliado en esta visita
        if (img && !hasZoomedCurrentSlideImage) {
            lightboxImg.src = img.src;
            lightboxImg.alt = img.alt;
            lightbox.classList.add('active');
            hasZoomedCurrentSlideImage = true;
            return;
        }

        // De lo contrario, avanzamos a la siguiente diapositiva
        goToSlide(currentSlide + 1);
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        
        // Si el lightbox está activo, lo cerramos e inmediatamente regresamos a la diapositiva anterior
        if (lightbox.classList.contains('active')) {
            lightbox.classList.remove('active');
            goToSlide(currentSlide - 1);
            return;
        }
        
        if (hasVerticalScroll(slide) && !isScrolledToTop(slide)) {
            slide.scrollBy({ top: -300, behavior: 'smooth' });
        } else {
            goToSlide(currentSlide - 1);
        }
    } else if (e.key === 'ArrowDown') {
        // Si el lightbox está activo, no hacemos scroll en la diapositiva
        if (lightbox.classList.contains('active')) return;
        
        e.preventDefault();
        if (hasVerticalScroll(slide)) {
            slide.scrollBy({ top: 300, behavior: 'smooth' });
        }
    } else if (e.key === 'ArrowUp') {
        // Si el lightbox está activo, no hacemos scroll en la diapositiva
        if (lightbox.classList.contains('active')) return;
        
        e.preventDefault();
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
            hasZoomedCurrentSlideImage = false;
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

// ========== LIGHTBOX ZOOM LOGIC ==========
const lightbox = document.getElementById('imageLightbox');
const lightboxImg = document.getElementById('lightboxImg');
const zoomableImages = document.querySelectorAll('.slide-content img');

zoomableImages.forEach(img => {
    img.addEventListener('click', (e) => {
        e.stopPropagation();
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt;
        lightbox.classList.add('active');
        hasZoomedCurrentSlideImage = true;
    });
});

// Cerrar al hacer clic en el overlay o botón de cierre
lightbox.addEventListener('click', () => {
    lightbox.classList.remove('active');
});

// Cerrar con Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox.classList.contains('active')) {
        lightbox.classList.remove('active');
    }
});

