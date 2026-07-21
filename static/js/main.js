/**
 * BookVerse Client-Side JavaScript
 * Handles real-time search autocomplete, UI debouncing, and interactivity.
 */

document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search_input');
    const autocompleteDropdown = document.getElementById('autocomplete_dropdown');
    let debounceTimer = null;

    if (searchInput && autocompleteDropdown) {
        // Listen to input changes for live autocomplete
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();

            clearTimeout(debounceTimer);
            if (query.length < 2) {
                autocompleteDropdown.innerHTML = '';
                autocompleteDropdown.classList.add('hidden');
                return;
            }

            // Debounce API call by 250ms
            debounceTimer = setTimeout(() => {
                fetchAutocompleteResults(query);
            }, 250);
        });

        // Hide dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !autocompleteDropdown.contains(e.target)) {
                autocompleteDropdown.classList.add('hidden');
            }
        });
    }

    async function fetchAutocompleteResults(query) {
        try {
            const response = await fetch(`/api/autocomplete?q=${encodeURIComponent(query)}`);
            if (!response.ok) return;

            const data = await response.json();
            const titles = data.titles || [];

            if (titles.length === 0) {
                autocompleteDropdown.innerHTML = '';
                autocompleteDropdown.classList.add('hidden');
                return;
            }

            autocompleteDropdown.innerHTML = titles.map(title => `
                <div class="autocomplete-item" data-title="${escapeHtml(title)}">
                    <i class="fa-solid fa-book"></i>
                    <span>${escapeHtml(title)}</span>
                </div>
            `).join('');

            autocompleteDropdown.classList.remove('hidden');

            // Attach click listeners to item options
            const items = autocompleteDropdown.querySelectorAll('.autocomplete-item');
            items.forEach(item => {
                item.addEventListener('click', () => {
                    const selectedTitle = item.getAttribute('data-title');
                    searchInput.value = selectedTitle;
                    autocompleteDropdown.classList.add('hidden');
                    document.getElementById('search_form').submit();
                });
            });
        } catch (err) {
            console.error('Autocomplete fetch error:', err);
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
