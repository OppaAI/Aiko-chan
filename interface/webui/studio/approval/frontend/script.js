// Set up filter click events once DOM loads
document.querySelectorAll('.filter-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const filter = tab.dataset.filter;
        currentFilter = filter;
        renderDraftList(filter);
    });
});

// Detect base path for API calls (studio is mounted at /studio/approval/)
const API_BASE = GraphBoot.apiBase();

let currentDraft = null;
let allDrafts = [];
let originalDraftText = '';
let currentSourceUrl = null;
let currentFilter = 'pending';
let selectedDraftDirs = new Set();
let bulkBusy = false;

function escapeHTML(str) {
    const d = document.createElement('div');
    d.textContent = str || '';
    return d.innerHTML;
}

function formatDate(dateStr) {
    if (!dateStr) return '—';
    try {
        const d = new Date(dateStr);
        return d.toLocaleString();
    } catch {
        return dateStr;
    }
}

async function loadDrafts() {
    try {
        const resp = await fetch(`${API_BASE}/drafts?status=all`);
        const data = await resp.json();
        allDrafts = data.drafts || [];
        const existingDirs = new Set(allDrafts.map(draft => draft.draft_dir));
        selectedDraftDirs = new Set([...selectedDraftDirs].filter(dir => existingDirs.has(dir)));
        renderDraftList(currentFilter);
        document.getElementById('header-status').textContent = `${allDrafts.length} drafts`;
        document.getElementById('draft-count').textContent = allDrafts.length;
    } catch (err) {
        console.error('Failed to load drafts:', err);
        document.getElementById('header-status').textContent = 'Failed to load';
    }
}

function filterDrafts(filter) {
    switch (filter) {
        case 'pending':
            return allDrafts.filter(d => !d.human_approved && !d.posted && !d.rejected);
        case 'approved':
            return allDrafts.filter(d => d.human_approved && !d.posted && !d.rejected);
        case 'posted':
            return allDrafts.filter(d => d.posted);
        case 'rejected':
            return allDrafts.filter(d => d.rejected);
        default:
            return allDrafts;
    }
}

function renderDraftList(filter) {
    const container = document.getElementById('drafts-list');
    const drafts = filterDrafts(filter);

    document.querySelectorAll('.filter-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });

    if (!drafts.length) {
        container.innerHTML = '<div style="color:var(--dim);font-size:12px;padding:20px;text-align:center">No drafts found</div>';
        updateBulkToolbar();
        return;
    }

    container.innerHTML = '';
    drafts.forEach(draft => {
        const row = document.createElement('div');
        row.className = 'draft-row';
        const div = document.createElement('div');
        const selected = selectedDraftDirs.has(draft.draft_dir);
        div.className = 'draft-item'
            + (currentDraft && currentDraft.draft_dir === draft.draft_dir ? ' active' : '')
            + (selected ? ' selected' : '');
        div.setAttribute('role', 'checkbox');
        div.setAttribute('tabindex', '0');
        div.setAttribute('aria-checked', String(selected));

        const statusBadges = [];
        if (draft.posted) {
            statusBadges.push('<span class="status-badge posted">Posted</span>');
        } else if (draft.rejected) {
            statusBadges.push('<span class="status-badge rejected">Rejected</span>');
        } else if (draft.human_approved) {
            statusBadges.push('<span class="status-badge approved">Approved</span>');
        } else {
            statusBadges.push('<span class="status-badge pending">Pending</span>');
        }
        if (draft.llm_enriched) {
            statusBadges.push('<span class="status-badge enriched">LLM Enriched</span>');
        }

        div.innerHTML = `
            <div class="draft-item-heading">
                <span class="draft-selection-mark" aria-hidden="true">${selected ? '✓' : ''}</span>
                <span class="draft-name">${escapeHTML(draft.relative_path)}</span>
            </div>
            <div class="draft-meta">
                <span>${formatDate(draft.created_at)}</span>
                <span>${escapeHTML(draft.category || 'general')}</span>
                <span>${escapeHTML(draft.provider)}</span>
            </div>
            <div class="draft-status">${statusBadges.join('')}</div>
        `;
        div.setAttribute('data-draft-dir', draft.draft_dir);
        const toggleSelection = () => {
            if (selectedDraftDirs.has(draft.draft_dir)) selectedDraftDirs.delete(draft.draft_dir);
            else selectedDraftDirs.add(draft.draft_dir);
            renderDraftList(currentFilter);
            const focusedItem = [...container.querySelectorAll('.draft-item')]
                .find(item => item.getAttribute('data-draft-dir') === draft.draft_dir);
            focusedItem?.focus();
        };
        div.onclick = toggleSelection;
        div.onkeydown = event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                toggleSelection();
            }
        };
        const reviewButton = document.createElement('button');
        reviewButton.type = 'button';
        reviewButton.className = 'draft-review-btn btn';
        reviewButton.textContent = 'Review';
        reviewButton.title = 'Open this draft without changing its selection';
        reviewButton.onclick = event => {
            event.stopPropagation();
            selectDraft(draft);
        };
        row.append(div, reviewButton);
        container.appendChild(row);
    });
    updateBulkToolbar();
}

function updateBulkToolbar() {
    const selected = allDrafts.filter(draft => selectedDraftDirs.has(draft.draft_dir));
    const pending = selected.filter(draft => !draft.human_approved && !draft.posted && !draft.rejected);
    const approved = selected.filter(draft => draft.human_approved && !draft.posted && !draft.rejected);
    const removable = selected.filter(draft => !draft.posted && !draft.rejected);
    const visible = filterDrafts(currentFilter);
    const visibleSelected = visible.filter(draft => selectedDraftDirs.has(draft.draft_dir)).length;
    const selectVisible = document.getElementById('select-visible-btn');
    if (!selectVisible) return;

    document.getElementById('selection-count').textContent = `${selected.length} selected${visibleSelected < selected.length ? ` · ${visibleSelected} visible` : ''}`;
    selectVisible.textContent = visible.length && visibleSelected === visible.length ? 'Deselect visible' : 'Select visible';
    document.getElementById('bulk-approve-btn').textContent = `Approve${pending.length ? ` (${pending.length})` : ''}`;
    document.getElementById('bulk-publish-btn').textContent = `Publish${approved.length ? ` (${approved.length})` : ''}`;
    document.getElementById('bulk-remove-btn').textContent = `Remove${removable.length ? ` (${removable.length})` : ''}`;
    document.getElementById('bulk-approve-btn').disabled = bulkBusy || pending.length === 0;
    document.getElementById('bulk-publish-btn').disabled = bulkBusy || approved.length === 0;
    document.getElementById('bulk-remove-btn').disabled = bulkBusy || removable.length === 0;
    selectVisible.disabled = bulkBusy || visible.length === 0;
    document.getElementById('clear-selection-btn').disabled = bulkBusy || selected.length === 0;
    document.querySelectorAll('#action-buttons button').forEach(button => {
        button.disabled = bulkBusy || button.dataset.requiresApproval === 'true';
    });
}

function selectDraft(draft, skipDirtyCheck = false, reloadContent = true) {
    if (!skipDirtyCheck && currentDraft && currentDraft.draft_dir !== draft.draft_dir && isContentDirty()) {
        const proceed = confirm('You have unsaved changes to this draft. Discard them and switch?');
        if (!proceed) return;
    }
    currentDraft = draft;
    document.querySelectorAll('.draft-item').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-draft-dir') === draft.draft_dir);
    });

    // Show info panel
    document.getElementById('info-empty').style.display = 'none';
    document.getElementById('info-content').style.display = 'block';

    // Populate info grid (Date field removed — Created already covers this)
    const grid = document.getElementById('info-grid');
    grid.innerHTML = `
        <div class="info-row">
            <div class="info-label">Path</div>
            <div class="info-value"><code>${escapeHTML(draft.relative_path)}</code></div>
        </div>
        <div class="info-row">
            <div class="info-label">Category</div>
            <div class="info-value">${escapeHTML(draft.category || 'general')}</div>
        </div>
        <div class="info-row">
            <div class="info-label">Provider</div>
            <div class="info-value">${escapeHTML(draft.provider)}</div>
        </div>
        <div class="info-row">
            <div class="info-label">Created</div>
            <div class="info-value">${formatDate(draft.created_at)}</div>
        </div>
        <div class="info-row">
            <div class="info-label">LLM Enriched</div>
            <div class="info-value" style="color: ${draft.llm_enriched ? 'var(--green)' : 'var(--dim)'}">${draft.llm_enriched ? 'Yes' : 'No'}</div>
        </div>
        <div class="info-row">
            <div class="info-label">Status</div>
            <div class="info-value">
                ${draft.posted ? '<span class="status-badge posted">Posted</span>' :
                  draft.rejected ? '<span class="status-badge rejected">Rejected</span>' :
                  draft.human_approved ? '<span class="status-badge approved">Approved</span>' :
                  '<span class="status-badge pending">Pending Review</span>'}
            </div>
        </div>
    `;

    // Action buttons in the exact order: Approve, Publish, Reject
    const actions = document.getElementById('action-buttons');
    if (draft.posted) {
        actions.innerHTML = '<span style="color: var(--dim); font-size: 12px;">This draft has already been posted</span>';
    } else if (draft.rejected) {
        actions.innerHTML = '<span style="color: var(--dim); font-size: 12px;">This draft has been rejected and archived</span>';
    } else if (draft.human_approved) {
        actions.innerHTML = `
            <button class="btn btn-warning" onclick="toggleApprove('${escapeHTML(draft.draft_dir)}', false)">Unapprove</button>
            <button class="btn btn-primary" onclick="postDraft('${escapeHTML(draft.draft_dir)}')">Publish</button>
            <button class="btn btn-danger" onclick="rejectDraft('${escapeHTML(draft.draft_dir)}')">Reject</button>
        `;
    } else {
        actions.innerHTML = `
            <button class="btn btn-success" onclick="toggleApprove('${escapeHTML(draft.draft_dir)}', true)">Approve</button>
            <button class="btn btn-primary" data-requires-approval="true" style="opacity: 0.5; cursor: not-allowed;" title="Approve before publishing" disabled>Publish</button>
            <button class="btn btn-danger" onclick="rejectDraft('${escapeHTML(draft.draft_dir)}')">Reject</button>
        `;
    }

    // Load full draft content unless this is a metadata-only status refresh.
    if (reloadContent) loadDraftContent(draft.draft_dir);
    updateBulkToolbar();
}

function autoGrowTextarea(el) {
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
}

function isContentDirty() {
    const contentEl = document.getElementById('content-text');
    if (contentEl.style.display === 'none') return false;
    return contentEl.value !== originalDraftText;
}

function updateEditStatus() {
    const contentEl = document.getElementById('content-text');
    const status = document.getElementById('edit-status');
    const dirty = isContentDirty();
    contentEl.classList.toggle('dirty', dirty);
    status.className = 'edit-status' + (dirty ? ' dirty' : '');
    status.textContent = dirty ? 'Unsaved changes' : '';
}

async function loadDraftContent(draftDir) {
    try {
        const resp = await fetch(`${API_BASE}/drafts/${encodeURIComponent(draftDir)}`);
        if (!resp.ok) throw new Error('Failed to load draft detail');
        const data = await resp.json();

        const contentEl = document.getElementById('content-text');
        const emptyEl = document.getElementById('content-empty');
        const editActions = document.getElementById('content-edit-actions');

        if (data.draft_text) {
            contentEl.value = data.draft_text;
            originalDraftText = data.draft_text;
            contentEl.style.display = 'block';
            editActions.style.display = 'flex';
            emptyEl.style.display = 'none';
            autoGrowTextarea(contentEl);
            updateEditStatus();
        } else {
            contentEl.value = '';
            originalDraftText = '';
            contentEl.style.display = 'none';
            editActions.style.display = 'none';
            emptyEl.style.display = 'flex';
            emptyEl.querySelector('p').textContent = 'No draft content available';
        }

        // Source webpage panel — rendered as Markdown (via MarkItDown on the backend)
        const posting = data.meta?.posting || data.posting;
        const urlContent = document.getElementById('url-content');
        const urlEmpty = document.getElementById('url-empty');
        const sourceTitle = document.getElementById('source-title');
        const urlMarkdown = document.getElementById('url-markdown');

        if (posting && posting.url) {
            urlEmpty.style.display = 'none';
            urlContent.style.display = 'flex';

            sourceTitle.textContent = posting.url;
            sourceTitle.style.textTransform = 'none';
            sourceTitle.style.letterSpacing = 'normal';
            sourceTitle.style.color = 'var(--orange)';
            sourceTitle.style.textDecoration = 'underline';
            sourceTitle.title = 'Open in a new tab';
            sourceTitle.onclick = () => window.open(posting.url, '_blank', 'noopener');
            sourceTitle.style.cursor = 'pointer';

            currentSourceUrl = posting.url;
            urlMarkdown.innerHTML = '<p style="color: var(--dim); font-size: 12px;">Loading markdown…</p>';

            fetch(`${API_BASE}/fetch-url?url=${encodeURIComponent(posting.url)}`)
                .then(resp => resp.json())
                .then(result => {
                    if (result.error) {
                        urlMarkdown.innerHTML = `<p style="color: var(--red); font-size: 12px;">Failed to load: ${escapeHTML(result.error)}</p>`;
                    } else if (typeof marked !== 'undefined') {
                        urlMarkdown.innerHTML = marked.parse(result.content || 'No content available');
                    } else {
                        // Fallback if marked.js failed to load — show raw markdown
                        urlMarkdown.textContent = result.content || 'No content available';
                    }
                })
                .catch(err => {
                    urlMarkdown.innerHTML = `<p style="color: var(--red); font-size: 12px;">Error loading URL: ${escapeHTML(err.message)}</p>`;
                });
        } else {
            urlContent.style.display = 'none';
            urlEmpty.style.display = 'flex';
            sourceTitle.textContent = 'Source Webpage';
            sourceTitle.style.textTransform = 'uppercase';
            sourceTitle.style.letterSpacing = '0.15em';
            sourceTitle.style.color = 'var(--orange)';
            sourceTitle.style.textDecoration = 'none';
            sourceTitle.onclick = null;
            sourceTitle.style.cursor = 'default';
            currentSourceUrl = null;
            urlMarkdown.innerHTML = '';
        }
    } catch (err) {
        console.error('Failed to load draft content:', err);
        document.getElementById('content-text').style.display = 'none';
        document.getElementById('content-edit-actions').style.display = 'none';
        document.getElementById('content-empty').style.display = 'flex';
        document.getElementById('content-empty').querySelector('p').textContent = 'Failed to load content';
    }
}

document.getElementById('content-text').addEventListener('input', (e) => {
    autoGrowTextarea(e.target);
    updateEditStatus();
});

document.getElementById('cancel-content-btn').addEventListener('click', () => {
    const contentEl = document.getElementById('content-text');
    contentEl.value = originalDraftText;
    autoGrowTextarea(contentEl);
    updateEditStatus();
});

document.getElementById('save-content-btn').addEventListener('click', async () => {
    if (!currentDraft) return;
    const contentEl = document.getElementById('content-text');
    const status = document.getElementById('edit-status');
    const saveBtn = document.getElementById('save-content-btn');
    const newText = contentEl.value;

    saveBtn.disabled = true;
    status.className = 'edit-status';
    status.textContent = 'Saving…';

    try {
        const resp = await fetch(`${API_BASE}/drafts/${encodeURIComponent(currentDraft.draft_dir)}/update-content`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ draft_text: newText }),
        });
        const data = await resp.json();
        if (!resp.ok || !data.success) {
            throw new Error(data.message || 'Save failed');
        }
        originalDraftText = newText;
        status.className = 'edit-status saved';
        status.textContent = 'Saved';
        contentEl.classList.remove('dirty');
        setTimeout(() => {
            if (status.textContent === 'Saved') status.textContent = '';
        }, 2500);
    } catch (err) {
        console.error('Failed to save draft content:', err);
        status.className = 'edit-status error';
        status.textContent = 'Failed to save';
    } finally {
        saveBtn.disabled = false;
    }
});

async function postDraft(draftDir) {
    try {
        const resp = await fetch(`${API_BASE}/drafts/${encodeURIComponent(draftDir)}/publish`, {
            method: 'POST',
        });
        const data = await resp.json();
        if (resp.ok && data.success) {
            loadDrafts();
            if (currentDraft && currentDraft.draft_dir === draftDir) {
                selectDraft({ ...currentDraft, posted: true, meta: data.meta });
            }
        } else {
            alert('Failed to publish: ' + (data.detail || data.message || 'Unknown error'));
        }
    } catch (err) {
        console.error('Publish failed:', err);
        alert('Failed to publish draft');
    }
}

async function rejectDraft(draftDir) {
    const confirmed = confirm(
        'This will move this draft to the rejected archive and remove it from the active list. Continue?'
    );
    if (!confirmed) return;

    try {
        const resp = await fetch(`${API_BASE}/drafts/${encodeURIComponent(draftDir)}/reject`, {
            method: 'POST',
        });
        const data = await resp.json();
        if (resp.ok && data.success) {
            if (currentDraft && currentDraft.draft_dir === draftDir) {
                currentDraft = null;
                document.getElementById('info-empty').style.display = 'flex';
                document.getElementById('info-content').style.display = 'none';
                document.getElementById('content-empty').style.display = 'flex';
                document.getElementById('content-empty').querySelector('p').textContent = 'Draft content will appear here';
                document.getElementById('content-text').style.display = 'none';
                document.getElementById('content-edit-actions').style.display = 'none';
                document.getElementById('url-content').style.display = 'none';
                document.getElementById('url-empty').style.display = 'flex';
            }
            loadDrafts();
        } else {
            alert('Failed to reject: ' + (data.detail || data.message || 'Unknown error'));
        }
    } catch (err) {
        console.error('Reject failed:', err);
        alert('Failed to reject draft');
    }
}

async function toggleApprove(draftDir, approve) {
    try {
        const resp = await fetch(`${API_BASE}/drafts/${encodeURIComponent(draftDir)}/toggle-approval`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ approve }),
        });
        const data = await resp.json();
        if (data.success) {
            loadDrafts();
            if (currentDraft && currentDraft.draft_dir === draftDir) {
                selectDraft({ ...currentDraft, human_approved: data.meta.human_approved, meta: data.meta });
            }
        } else {
            alert('Failed to update draft: ' + (data.message || 'Unknown error'));
        }
    } catch (err) {
        console.error('Toggle approval failed:', err);
        alert('Failed to update draft approval status');
    }
}

function captureOpenDraftState() {
    const content = document.getElementById('content-text');
    return {
        draftDir: currentDraft?.draft_dir || null,
        content: content.value,
        original: originalDraftText,
        contentDisplay: content.style.display,
        sourceUrl: currentSourceUrl,
    };
}

function openDraftStateUnchanged(before, after) {
    return before.draftDir === after.draftDir
        && before.content === after.content
        && before.original === after.original
        && before.contentDisplay === after.contentDisplay
        && before.sourceUrl === after.sourceUrl;
}

function clearOpenDraft() {
    currentDraft = null;
    document.getElementById('info-empty').style.display = 'flex';
    document.getElementById('info-content').style.display = 'none';
    document.getElementById('content-empty').style.display = 'flex';
    document.getElementById('content-empty').querySelector('p').textContent = 'Draft content will appear here';
    document.getElementById('content-text').style.display = 'none';
    document.getElementById('content-edit-actions').style.display = 'none';
    document.getElementById('url-content').style.display = 'none';
    document.getElementById('url-empty').style.display = 'flex';
    currentSourceUrl = null;
    originalDraftText = '';
    updateBulkToolbar();
}

async function runBulkAction(action) {
    const selected = allDrafts.filter(draft => selectedDraftDirs.has(draft.draft_dir));
    const eligible = selected.filter(draft => {
        if (draft.posted || draft.rejected) return false;
        if (action === 'approve') return !draft.human_approved;
        if (action === 'publish') return draft.human_approved;
        return action === 'reject';
    });
    if (!eligible.length || bulkBusy) return;

    const labels = { approve: 'approve', publish: 'publish to Threads', reject: 'move to the rejected archive' };
    const progressLabels = { approve: 'Approving', publish: 'Publishing', reject: 'Removing' };
    const verb = labels[action];
    if (action === 'publish' && !confirm(`Publish ${eligible.length} approved post${eligible.length === 1 ? '' : 's'} to Threads now?`)) return;
    if (action === 'reject' && !confirm(`Remove ${eligible.length} post${eligible.length === 1 ? '' : 's'} from the active list and move them to the rejected archive?`)) return;
    if (action === 'approve' && !confirm(`Approve ${eligible.length} selected post${eligible.length === 1 ? '' : 's'}?`)) return;
    if (currentDraft && eligible.some(draft => draft.draft_dir === currentDraft.draft_dir) && isContentDirty()) {
        if (!confirm('The selected draft has unsaved changes. Continue without saving them?')) return;
    }

    const openDraftAtStart = captureOpenDraftState();
    bulkBusy = true;
    const status = document.getElementById('bulk-status');
    status.textContent = `${progressLabels[action]} 0/${eligible.length} posts…`;
    updateBulkToolbar();
    try {
        const results = [];
        let stoppedAfterFailure = false;
        for (let offset = 0; offset < eligible.length; offset += 100) {
            const chunk = eligible.slice(offset, offset + 100);
            const batchNumber = Math.floor(offset / 100) + 1;
            const batchCount = Math.ceil(eligible.length / 100);
            status.textContent = `${progressLabels[action]} batch ${batchNumber}/${batchCount} (${chunk.length} posts)…`;
            try {
                const response = await fetch(`${API_BASE}/drafts/bulk-action`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action, draft_dirs: chunk.map(draft => draft.draft_dir) }),
                });
                const data = await response.json();
                if (!response.ok) {
                    chunk.forEach(draft => results.push({ draft_dir: draft.draft_dir, success: false, detail: data.detail || 'Bulk action failed' }));
                    stoppedAfterFailure = true;
                    break;
                }
                const chunkResults = new Map((data.results || []).map(result => [result.draft_dir, result]));
                chunk.forEach(draft => results.push(chunkResults.get(draft.draft_dir) || {
                    draft_dir: draft.draft_dir,
                    success: false,
                    detail: 'The server did not return a result for this draft',
                }));
            } catch (err) {
                chunk.forEach(draft => results.push({
                    draft_dir: draft.draft_dir,
                    success: false,
                    detail: `No response received; verify status before retrying (${err.message})`,
                }));
                stoppedAfterFailure = true;
                break;
            }
        }
        if (stoppedAfterFailure) {
            const sentCount = results.length;
            eligible.slice(sentCount).forEach(draft => results.push({
                draft_dir: draft.draft_dir,
                success: false,
                detail: 'Not sent because the previous batch did not complete reliably',
            }));
        }

        const succeeded = results.filter(result => result.success);
        const failed = results.filter(result => !result.success);
        succeeded.forEach(result => selectedDraftDirs.delete(result.draft_dir));
        await loadDrafts();

        // Only refresh metadata if the reviewer is still looking at the same
        // draft and its editor has not changed while the bulk request ran.
        const openDraftAfterRequest = captureOpenDraftState();
        if (openDraftAtStart.draftDir && openDraftStateUnchanged(openDraftAtStart, openDraftAfterRequest)) {
            const openDraftWasRemoved = action === 'reject'
                && succeeded.some(result => result.draft_dir === openDraftAtStart.draftDir);
            if (openDraftWasRemoved) {
                clearOpenDraft();
            } else if (succeeded.some(result => result.draft_dir === openDraftAtStart.draftDir)) {
                const refreshedDraft = allDrafts.find(draft => draft.draft_dir === openDraftAtStart.draftDir);
                if (refreshedDraft) selectDraft(refreshedDraft, true, false);
            }
        }
        const past = succeeded.length === 1 ? 'post' : 'posts';
        status.textContent = `${succeeded.length}/${eligible.length} ${past} ${action === 'reject' ? 'removed' : action === 'publish' ? 'published' : 'approved'}${failed.length ? ` · ${failed.length} failed` : ''}`;
        if (failed.length) {
            status.title = failed.map(result => {
                const draft = eligible.find(item => item.draft_dir === result.draft_dir);
                return `${draft?.relative_path || 'Draft'}: ${result.detail || result.message || 'Failed'}`;
            }).join('\n');
        } else {
            status.removeAttribute('title');
        }
    } catch (err) {
        console.error('Bulk approval action failed:', err);
        status.textContent = `Failed to ${verb}: ${err.message}`;
    } finally {
        bulkBusy = false;
        updateBulkToolbar();
    }
}

document.getElementById('select-visible-btn').addEventListener('click', () => {
    const visible = filterDrafts(currentFilter);
    const allVisibleSelected = visible.length > 0 && visible.every(draft => selectedDraftDirs.has(draft.draft_dir));
    visible.forEach(draft => {
        if (allVisibleSelected) selectedDraftDirs.delete(draft.draft_dir);
        else selectedDraftDirs.add(draft.draft_dir);
    });
    renderDraftList(currentFilter);
});
document.getElementById('clear-selection-btn').addEventListener('click', () => {
    selectedDraftDirs.clear();
    renderDraftList(currentFilter);
});
document.getElementById('bulk-approve-btn').addEventListener('click', () => runBulkAction('approve'));
document.getElementById('bulk-publish-btn').addEventListener('click', () => runBulkAction('publish'));
document.getElementById('bulk-remove-btn').addEventListener('click', () => runBulkAction('reject'));

document.getElementById('refresh-btn').addEventListener('click', loadDrafts);

// Initial load
loadDrafts();
setInterval(loadDrafts, 30000);
