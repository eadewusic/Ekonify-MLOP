// Counter animation function
function animateCounter(elementId, start, end, duration, prefix = '', suffix = '') {
    const element = document.getElementById(elementId);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        
        // Use different formats based on the number size
        let currentValue = Math.floor(progress * (end - start) + start);
        
        if (end >= 1000000) {
            // Format as "1M" for millions
            element.textContent = prefix + (currentValue / 1000000).toFixed(1).replace(/\.0$/, '') + 'M' + suffix;
        } else if (end >= 1000) {
            // Format as "500" for hundreds
            element.textContent = prefix + currentValue + suffix;
        } else {
            // Format as regular number with possible decimal point
            element.textContent = prefix + currentValue + suffix;
        }
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Function to check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Function to start animations when section is in view
function handleScroll() {
    const impactSection = document.querySelector('.impact');
    if (isInViewport(impactSection) && !window.countersActivated) {
        // Start counter animations
        animateCounter('items-counter', 0, 1000000, 2000); // 1M
        animateCounter('tons-counter', 0, 500, 2000); // 500
        animateCounter('accuracy-counter', 0, 75, 2000); // 75%
        
        // Footer counters
        animateCounter('total-classified', 0, 1234567, 2000, '', ',');
        animateCounter('total-impact', 0, 567, 2000);
        
        // Set flag to prevent re-triggering
        window.countersActivated = true;
        
        // Remove scroll listener once activated
        window.removeEventListener('scroll', handleScroll);
    }
}

// Add scroll event listener
window.addEventListener('scroll', handleScroll);

// Also check once on load (in case the section is already visible)
document.addEventListener('DOMContentLoaded', handleScroll);