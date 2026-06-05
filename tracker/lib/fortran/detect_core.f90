! =============================================================
! detect_core.f90 - f2py-callable Fortran subroutines
! T1.0: real (single precision) to match original Fortran code
!
! Pure computational routines with no I/O or module dependencies.
! Build: f2py -c -m detect_core detect_core.f90
! =============================================================


subroutine calc_vorticity(ua, va, vor, lon, lat, nlon, nlat, ntime)
    ! Calculate relative vorticity from u/v winds on a regular grid.
    ! Handles periodic longitude boundaries.
    !
    !f2py intent(in)  :: ua, va, lon, lat, nlon, nlat, ntime
    !f2py intent(out) :: vor
    implicit none
    integer, intent(in) :: nlon, nlat, ntime
    real, intent(in)  :: ua(nlon, nlat, ntime)
    real, intent(in)  :: va(nlon, nlat, ntime)
    real, intent(in)  :: lon(nlon), lat(nlat)
    real, intent(out) :: vor(nlon, nlat, ntime)

    real, parameter :: pi = 3.141592
    real, parameter :: R  = 6371000.
    real :: dx, dy
    integer :: i, j, k
    integer :: ip1, im1

    vor = 0.

    do k = 1, ntime
        do j = 2, nlat - 1
            dy = (lat(j+1) - lat(j-1)) * 2. * R * pi / 360.
            do i = 1, nlon
                ! periodic longitude
                ip1 = i + 1; if (ip1 > nlon) ip1 = 1
                im1 = i - 1; if (im1 < 1)    im1 = nlon

                dx = (lon(ip1) - lon(im1))
                if (dx < 0.) dx = dx + 360.
                if (dx > 360.) dx = dx - 360.
                dx = dx * 2. * R * pi / 360. * cos(lat(j) * pi / 180.)

                if (abs(dx) > 1.) then
                    vor(i, j, k) = (va(ip1, j, k) - va(im1, j, k)) / dx &
                                 - (ua(i, j+1, k) - ua(i, j-1, k)) / dy
                endif
            enddo
        enddo
    enddo

end subroutine calc_vorticity


subroutine find_centers(vor, psl, wspd, lon, lat, &
                        nlon, nlat, ntime, &
                        cri_vor, cri_lat, maxd, &
                        m_search, psl_search_km, wspd_search_km, &
                        out_lon, out_lat, out_psl, out_wspd, &
                        out_vlon, out_vlat, &
                        ncenter, max_centers, near_ocean)
    ! Find disturbance centers for all timesteps.
    ! For each timestep, identifies vorticity extrema co-located with pressure minima.
    !
    !f2py intent(in)  :: vor, psl, wspd, lon, lat
    !f2py intent(in)  :: nlon, nlat, ntime, cri_vor, cri_lat, maxd
    !f2py intent(in)  :: m_search, psl_search_km, wspd_search_km, max_centers
    !f2py intent(in)  :: near_ocean
    !f2py intent(out) :: out_lon, out_lat, out_psl, out_wspd, out_vlon, out_vlat, ncenter
    implicit none
    integer, intent(in) :: nlon, nlat, ntime, maxd, max_centers, m_search
    real, intent(in) :: vor(nlon, nlat, ntime)
    real, intent(in) :: psl(nlon, nlat, ntime)
    real, intent(in) :: wspd(nlon, nlat, ntime)
    real, intent(in) :: lon(nlon), lat(nlat)
    real, intent(in) :: cri_vor, cri_lat
    real, intent(in) :: psl_search_km, wspd_search_km
    integer, intent(in) :: near_ocean(nlon, nlat)

    real, intent(out) :: out_lon(ntime, max_centers)
    real, intent(out) :: out_lat(ntime, max_centers)
    real, intent(out) :: out_psl(ntime, max_centers)
    real, intent(out) :: out_wspd(ntime, max_centers)
    real, intent(out) :: out_vlon(ntime, max_centers)
    real, intent(out) :: out_vlat(ntime, max_centers)
    integer, intent(out) :: ncenter(ntime)

    ! Local - allocatable to avoid stack overflow on large grids
    real, allocatable :: vor2(:,:), psl2(:,:), wspd2(:,:)
    ! PSL local minimum cache: 0=unchecked, 1=is_min, -1=not_min
    integer, allocatable :: psl_min_cache(:,:)
    ! Precomputed distance table: dist_by_lat(jc, di, dj) = haversine from
    ! lat(jc) to lat(jc+dj) with longitude offset di.  On a regular grid,
    ! this is independent of the absolute longitude index.
    real, allocatable :: dist_by_lat(:,:,:)

    real :: vormax, vormin, pslmin, distkm_val
    integer :: i, j, k, ii, jj, m, ll, ik, jk, iw, idx
    integer :: ne, halo, iref, iwrap, jc, di, dj
    logical :: is_extremum
    ! Distance-sorted offset arrays per latitude (for early exit in PSL search)
    integer :: max_offsets
    integer, allocatable :: sorted_di(:,:), sorted_dj(:,:), sorted_count(:)
    real, allocatable    :: sorted_dist(:,:)
    ! Temp arrays for sorting
    real, allocatable    :: tmp_dists(:)
    integer, allocatable :: tmp_di_arr(:), tmp_dj_arr(:)
    integer :: n_valid, si, sj
    real :: tmp_swap

    ! 2-pass candidate collection arrays (sized relative to max_centers)
    integer :: max_cand
    integer, allocatable :: cand_vi(:), cand_vj(:)
    integer, allocatable :: cand_pi(:), cand_pj(:), cand_pii(:)
    real, allocatable    :: cand_dist(:)
    integer :: ncand, ic, jc_idx, best_ic
    real    :: best_cdist
    integer, allocatable :: cand_assigned_to(:)
    ! Temp variables for best PSL search within ring
    integer :: tmp_iw, tmp_jj, tmp_ii
    integer :: n_cand_cap_hit  ! count timesteps where max_cand was exceeded

    ne = maxd * 2
    halo = max(m_search + maxd, 2 * m_search)  ! enough for wind search around PSL center
    max_cand = max_centers * 4
    out_lon  = -999.
    out_lat  = -999.
    out_psl  = -999.
    out_wspd = -999.
    out_vlon = -999.
    out_vlat = -999.
    ncenter  = 0

    ! --- Precompute distance table (once, before OMP loop) ---
    allocate(dist_by_lat(nlat, -m_search:m_search, -m_search:m_search))
    dist_by_lat = 9999.
    iref = nlon / 2 + 1
    do jc = 1, nlat
        do di = -m_search, m_search
            iwrap = iref + di
            if (iwrap < 1) iwrap = iwrap + nlon
            if (iwrap > nlon) iwrap = iwrap - nlon
            do dj = -m_search, m_search
                if (jc + dj >= 1 .and. jc + dj <= nlat) then
                    call haversine_km(lon(iref), lat(jc), lon(iwrap), lat(jc + dj), &
                                     dist_by_lat(jc, di, dj))
                endif
            enddo
        enddo
    enddo

    ! --- Precompute distance-sorted offset lists per latitude ---
    ! All (di, dj) offsets within m_search box, sorted by true km distance.
    ! Enables early exit in PSL search when dist > best_cdist.
    max_offsets = (2 * m_search + 1) ** 2
    allocate(sorted_di(nlat, max_offsets))
    allocate(sorted_dj(nlat, max_offsets))
    allocate(sorted_dist(nlat, max_offsets))
    allocate(sorted_count(nlat))
    allocate(tmp_dists(max_offsets), tmp_di_arr(max_offsets), tmp_dj_arr(max_offsets))

    do jc = 1, nlat
        n_valid = 0
        do dj = -m_search, m_search
            if (jc + dj < 1 .or. jc + dj > nlat) cycle
            do di = -m_search, m_search
                if (dist_by_lat(jc, di, dj) <= psl_search_km) then
                    n_valid = n_valid + 1
                    tmp_di_arr(n_valid) = di
                    tmp_dj_arr(n_valid) = dj
                    tmp_dists(n_valid) = dist_by_lat(jc, di, dj)
                endif
            enddo
        enddo
        ! Simple insertion sort (max_offsets is small, ~hundreds)
        do si = 2, n_valid
            tmp_swap = tmp_dists(si)
            di = tmp_di_arr(si)
            dj = tmp_dj_arr(si)
            sj = si - 1
            do while (sj >= 1 .and. tmp_dists(sj) > tmp_swap)
                tmp_dists(sj + 1) = tmp_dists(sj)
                tmp_di_arr(sj + 1) = tmp_di_arr(sj)
                tmp_dj_arr(sj + 1) = tmp_dj_arr(sj)
                sj = sj - 1
            enddo
            tmp_dists(sj + 1) = tmp_swap
            tmp_di_arr(sj + 1) = di
            tmp_dj_arr(sj + 1) = dj
        enddo
        sorted_count(jc) = n_valid
        sorted_di(jc, 1:n_valid) = tmp_di_arr(1:n_valid)
        sorted_dj(jc, 1:n_valid) = tmp_dj_arr(1:n_valid)
        sorted_dist(jc, 1:n_valid) = tmp_dists(1:n_valid)
    enddo
    deallocate(tmp_dists, tmp_di_arr, tmp_dj_arr)

    n_cand_cap_hit = 0
    !$OMP PARALLEL PRIVATE(vor2, psl2, wspd2, psl_min_cache, &
    !$OMP&  ll, i, j, ii, jj, ik, jk, iw, idx, &
    !$OMP&  is_extremum, vormax, vormin, pslmin, distkm_val, &
    !$OMP&  ncand, cand_vi, cand_vj, cand_pi, cand_pj, cand_pii, cand_dist, &
    !$OMP&  ic, jc_idx, best_ic, best_cdist, cand_assigned_to, &
    !$OMP&  tmp_iw, tmp_jj, tmp_ii)
    allocate(vor2(1-halo:nlon+halo, nlat))
    allocate(psl2(1-halo:nlon+halo, nlat))
    allocate(wspd2(1-halo:nlon+halo, nlat))
    allocate(psl_min_cache(nlon, nlat))
    allocate(cand_vi(max_cand), cand_vj(max_cand))
    allocate(cand_pi(max_cand), cand_pj(max_cand), cand_pii(max_cand))
    allocate(cand_dist(max_cand), cand_assigned_to(max_cand))
    !$OMP DO
    do k = 1, ntime

        ! Extend longitude for periodic boundary
        vor2(1:nlon, :) = vor(:, :, k)
        psl2(1:nlon, :) = psl(:, :, k)
        wspd2(1:nlon, :) = wspd(:, :, k)
        do i = 1, halo
            vor2(1 - i, :)     = vor(nlon + 1 - i, :, k)
            vor2(nlon + i, :)  = vor(i, :, k)
            psl2(1 - i, :)     = psl(nlon + 1 - i, :, k)
            psl2(nlon + i, :)  = psl(i, :, k)
            wspd2(1 - i, :)    = wspd(nlon + 1 - i, :, k)
            wspd2(nlon + i, :) = wspd(i, :, k)
        enddo

        psl_min_cache = 0
        ncand = 0

        ! === Pass 1: collect all (vorticity extremum → nearest PSL minimum) candidates ===
        do j = 1 + ne, nlat - ne
            if (abs(lat(j)) > cri_lat) cycle

            do i = 1, nlon
                if (near_ocean(i, j) == 0) cycle
                if (abs(vor2(i, j)) < cri_vor) cycle

                is_extremum = .false.
                if (lat(j) >= 0.) then
                    vormax = maxval(vor2(i-maxd:i+maxd, j-maxd:j+maxd))
                    if (vor2(i, j) == vormax) is_extremum = .true.
                else
                    vormin = minval(vor2(i-maxd:i+maxd, j-maxd:j+maxd))
                    if (vor2(i, j) == vormin) is_extremum = .true.
                endif
                if (.not. is_extremum) cycle

                ! Find nearest PSL minimum by true km distance.
                ! Offsets are pre-sorted by distance, so once we pass
                ! best_cdist, no further offset can improve — early exit.
                best_cdist = psl_search_km + 1.0
                best_ic = 0  ! reuse as temp: 0 = no candidate yet
                do idx = 1, sorted_count(j)
                    distkm_val = sorted_dist(j, idx)
                    if (distkm_val > best_cdist) exit  ! early exit: sorted, no better candidate possible

                    ii = i + sorted_di(j, idx)
                    jj = j + sorted_dj(j, idx)
                    iw = ii
                    if (iw < 1) iw = iw + nlon
                    if (iw > nlon) iw = iw - nlon

                    ! PSL local min check with cache
                    if (psl_min_cache(iw, jj) == 0) then
                        pslmin = minval(psl2(ii-maxd:ii+maxd, max(jj-maxd,1):min(jj+maxd,nlat)))
                        if (psl2(ii, jj) == pslmin) then
                            psl_min_cache(iw, jj) = 1
                        else
                            psl_min_cache(iw, jj) = -1
                        endif
                    endif
                    if (psl_min_cache(iw, jj) == 1) then
                        if (best_ic == 0) then
                            ! First PSL minimum found
                            best_cdist = distkm_val
                            best_ic = 1
                            tmp_iw = iw
                            tmp_jj = jj
                            tmp_ii = ii
                        else if (distkm_val < best_cdist .or. &
                                 (distkm_val == best_cdist .and. &
                                  psl2(ii, jj) < psl2(tmp_ii, tmp_jj))) then
                            ! Same distance but lower PSL (can't be closer since sorted)
                            best_cdist = distkm_val
                            tmp_iw = iw
                            tmp_jj = jj
                            tmp_ii = ii
                        endif
                    endif
                enddo
                ! Commit the best candidate if found
                if (best_ic > 0) then
                    if (ncand < max_cand) then
                        ncand = ncand + 1
                        cand_vi(ncand) = i
                        cand_vj(ncand) = j
                        cand_pi(ncand) = tmp_iw
                        cand_pj(ncand) = tmp_jj
                        cand_pii(ncand) = tmp_ii
                        cand_dist(ncand) = best_cdist
                    endif
                endif

            enddo
        enddo

        ! Track if this timestep hit the cap
        if (ncand >= max_cand) then
            !$OMP ATOMIC
            n_cand_cap_hit = n_cand_cap_hit + 1
        endif

        ! === Pass 2: resolve conflicts — same PSL claimed by multiple vorticity extrema ===
        ! For each PSL point, keep only the candidate with smallest distance.
        ! Tie-break: stronger vorticity (larger |vor|) wins.
        cand_assigned_to = 0

        ! Simple O(n^2) conflict resolution — ncand is small (~tens)
        do ic = 1, ncand
            best_ic = ic
            best_cdist = cand_dist(ic)
            do jc_idx = 1, ncand
                if (jc_idx == ic) cycle
                if (cand_pi(jc_idx) == cand_pi(ic) .and. cand_pj(jc_idx) == cand_pj(ic)) then
                    ! Same PSL point — keep closer; tie-break: stronger vor, then lower index
                    if (cand_dist(jc_idx) < best_cdist .or. &
                        (cand_dist(jc_idx) == best_cdist .and. &
                         abs(vor(cand_vi(jc_idx), cand_vj(jc_idx), k)) > &
                         abs(vor(cand_vi(best_ic), cand_vj(best_ic), k))) .or. &
                        (cand_dist(jc_idx) == best_cdist .and. &
                         abs(vor(cand_vi(jc_idx), cand_vj(jc_idx), k)) == &
                         abs(vor(cand_vi(best_ic), cand_vj(best_ic), k)) .and. &
                         jc_idx < best_ic)) then
                        best_ic = jc_idx
                        best_cdist = cand_dist(jc_idx)
                    endif
                endif
            enddo
            if (best_ic == ic) then
                cand_assigned_to(ic) = 1  ! this candidate wins
            endif
        enddo

        ! === Emit winning candidates as centers ===
        ll = 0
        do ic = 1, ncand
            if (cand_assigned_to(ic) /= 1) cycle
            if (ll >= max_centers) exit

            i = cand_vi(ic)
            j = cand_vj(ic)
            iw = cand_pi(ic)
            jj = cand_pj(ic)
            ii = cand_pii(ic)

            ll = ll + 1
            out_lon(k, ll) = lon(iw)
            out_lat(k, ll) = lat(jj)
            out_psl(k, ll) = psl2(ii, jj)

            ! Max wind speed within wspd_search_km
            out_wspd(k, ll) = -999.
            do ik = -m_search, m_search
                do jk = -m_search, m_search
                    if (jj+jk >= 1 .and. jj+jk <= nlat) then
                        if (dist_by_lat(jj, ik, jk) <= wspd_search_km) then
                            if (wspd2(ii+ik, jj+jk) > out_wspd(k, ll)) then
                                out_wspd(k, ll) = wspd2(ii+ik, jj+jk)
                            endif
                        endif
                    endif
                enddo
            enddo

            out_vlon(k, ll) = lon(i)
            out_vlat(k, ll) = lat(j)
        enddo

        ncenter(k) = ll
    enddo
    !$OMP END DO
    deallocate(vor2, psl2, wspd2, psl_min_cache)
    deallocate(cand_vi, cand_vj, cand_pi, cand_pj, cand_pii, cand_dist, cand_assigned_to)
    !$OMP END PARALLEL
    deallocate(dist_by_lat, sorted_di, sorted_dj, sorted_dist, sorted_count)

    if (n_cand_cap_hit > 0) then
        write(*,'(A,I5,A,I5,A)') '  WARNING: max_cand=', max_cand, &
            ' hit in ', n_cand_cap_hit, ' timesteps'
    endif

end subroutine find_centers


subroutine connect_tracks(clon, clat, cpsl, cwspd, ncenter, &
                          ntime, max_centers, mindist_km, mindhr, dt_hours, &
                          mask, lon, lat, nlon, nlat, oceanid, cri_gen_lat, &
                          track_lon, track_lat, track_psl, track_wspd, &
                          track_time_idx, track_lengths, ntrack, max_tracks, max_tracklen)
    ! Connect detected centers across timesteps into tracks.
    ! Per-timestep 2-pass: all active tracks advance simultaneously,
    ! conflicts resolved by nearest distance.
    !
    !f2py intent(in)  :: clon, clat, cpsl, cwspd, ncenter
    !f2py intent(in)  :: ntime, max_centers, mindist_km, mindhr, dt_hours
    !f2py intent(in)  :: mask, lon, lat, nlon, nlat, oceanid, cri_gen_lat
    !f2py intent(in)  :: max_tracks, max_tracklen
    !f2py intent(out) :: track_lon, track_lat, track_psl, track_wspd, track_time_idx, track_lengths, ntrack
    implicit none
    integer, intent(in) :: ntime, max_centers, nlon, nlat
    integer, intent(in) :: max_tracks, max_tracklen
    real, intent(in) :: clon(ntime, max_centers)
    real, intent(in) :: clat(ntime, max_centers)
    real, intent(in) :: cpsl(ntime, max_centers)
    real, intent(in) :: cwspd(ntime, max_centers)
    integer, intent(in) :: ncenter(ntime)
    real, intent(in) :: mindist_km, oceanid, cri_gen_lat
    integer, intent(in) :: mindhr, dt_hours
    real, intent(in) :: mask(nlon, nlat), lon(nlon), lat(nlat)

    real, intent(out) :: track_lon(max_tracks, max_tracklen)
    real, intent(out) :: track_lat(max_tracks, max_tracklen)
    real, intent(out) :: track_psl(max_tracks, max_tracklen)
    real, intent(out) :: track_wspd(max_tracks, max_tracklen)
    integer, intent(out) :: track_time_idx(max_tracks, max_tracklen)
    integer, intent(out) :: track_lengths(max_tracks)
    integer, intent(out) :: ntrack

    ! Local
    integer :: used(ntime, max_centers)  ! 1 if center is already part of a track
    integer :: k, l, ll, it, ia, ib, slot
    real :: distkm_val
    integer :: ix(1), iy(1)

    ! Active track management
    integer :: n_active                          ! number of currently active tracks
    integer :: act_track(max_tracks)             ! which output track slot
    integer :: act_k(max_tracks)                 ! current timestep of head
    integer :: act_l(max_tracks)                 ! current center index of head

    ! Per-timestep matching
    integer :: want_ll(max_tracks)               ! which k+1 center each active track wants
    real    :: want_dist(max_tracks)             ! distance to wanted center
    integer :: winner(max_centers)               ! for each k+1 center, which active track wins
    real    :: winner_dist(max_centers)           ! winning distance
    logical :: alive(max_tracks)                 ! still active after conflict resolution

    ! Free slot recycling: terminated tracks that fail mindhr release their
    ! slot so it can be reused.  This prevents ntrack from growing unboundedly
    ! with short-lived disturbances.
    integer :: n_free
    integer :: free_slots(max_tracks)

    used = 0
    track_lon = -999.
    track_lat = -999.
    track_psl = -999.
    track_wspd = -999.
    track_time_idx = 0
    track_lengths = 0
    ntrack = 0
    n_active = 0
    n_free = 0

    do k = 1, ntime
        ! === Step 1: Start new tracks from unused centers at timestep k ===
        do l = 1, ncenter(k)
            if (clon(k, l) < -900.) cycle
            if (used(k, l) == 1) cycle

            ! Check ocean + genesis latitude
            ix = minloc(abs(lon - clon(k, l)))
            iy = minloc(abs(lat - clat(k, l)))
            if (mask(ix(1), iy(1)) /= oceanid) cycle
            if (abs(clat(k, l)) > cri_gen_lat) cycle

            ! Allocate a track slot: reuse freed slot or allocate new
            if (n_free > 0) then
                slot = free_slots(n_free)
                n_free = n_free - 1
            else
                if (ntrack + 1 > max_tracks) then
                    write(*,'(A,I8,A,I8,A)') &
                        '  ERROR: max_tracks=', max_tracks, &
                        ' reached at timestep ', k, &
                        ' — connect_tracks terminated early'
                    goto 9000
                endif
                ntrack = ntrack + 1
                slot = ntrack
            endif

            track_lon(slot, 1) = clon(k, l)
            track_lat(slot, 1) = clat(k, l)
            track_psl(slot, 1) = cpsl(k, l)
            track_wspd(slot, 1) = cwspd(k, l)
            track_time_idx(slot, 1) = k
            track_lengths(slot) = 1
            used(k, l) = 1

            ! Add to active list
            n_active = n_active + 1
            act_track(n_active) = slot
            act_k(n_active) = k
            act_l(n_active) = l
        enddo

        if (k == ntime) exit
        if (n_active == 0) cycle

        ! === Step 2: Each active track finds its nearest center at k+1 ===
        do ia = 1, n_active
            want_ll(ia) = 0
            want_dist(ia) = 99999.0
            do ll = 1, ncenter(k + 1)
                if (clon(k + 1, ll) < -900.) cycle
                call haversine_km(clon(act_k(ia), act_l(ia)), clat(act_k(ia), act_l(ia)), &
                                  clon(k + 1, ll), clat(k + 1, ll), distkm_val)
                if (distkm_val <= mindist_km .and. &
                    (distkm_val < want_dist(ia) .or. &
                     (distkm_val == want_dist(ia) .and. want_ll(ia) > 0 .and. &
                      cpsl(k + 1, ll) < cpsl(k + 1, want_ll(ia))))) then
                    want_dist(ia) = distkm_val
                    want_ll(ia) = ll
                endif
            enddo
        enddo

        ! === Step 3: Resolve conflicts — same k+1 center wanted by multiple tracks ===
        ! For each k+1 center, keep only the closest track.
        ! Tie-break: longer track wins; if still tied, lower PSL (stronger) wins.
        winner = 0
        winner_dist = 99999.0
        do ia = 1, n_active
            ll = want_ll(ia)
            if (ll == 0) cycle
            it = act_track(ia)
            if (want_dist(ia) < winner_dist(ll) .or. &
                (want_dist(ia) == winner_dist(ll) .and. winner(ll) > 0 .and. &
                 (track_lengths(it) > track_lengths(act_track(winner(ll))) .or. &
                  (track_lengths(it) == track_lengths(act_track(winner(ll))) .and. &
                   cpsl(act_k(ia), act_l(ia)) < cpsl(act_k(winner(ll)), act_l(winner(ll)))) .or. &
                  (track_lengths(it) == track_lengths(act_track(winner(ll))) .and. &
                   cpsl(act_k(ia), act_l(ia)) == cpsl(act_k(winner(ll)), act_l(winner(ll))) .and. &
                   ia < winner(ll))))) then
                winner(ll) = ia
                winner_dist(ll) = want_dist(ia)
            endif
        enddo

        ! === Step 4: Advance winners, terminate losers ===
        alive = .false.
        do ia = 1, n_active
            ll = want_ll(ia)
            if (ll > 0 .and. winner(ll) == ia) then
                ! This track wins — extend it
                it = act_track(ia)
                track_lengths(it) = track_lengths(it) + 1
                if (track_lengths(it) > max_tracklen) then
                    track_lengths(it) = track_lengths(it) - 1
                    cycle  ! terminate — max length reached
                endif
                track_lon(it, track_lengths(it)) = clon(k + 1, ll)
                track_lat(it, track_lengths(it)) = clat(k + 1, ll)
                track_psl(it, track_lengths(it)) = cpsl(k + 1, ll)
                track_wspd(it, track_lengths(it)) = cwspd(k + 1, ll)
                track_time_idx(it, track_lengths(it)) = k + 1
                used(k + 1, ll) = 1

                act_k(ia) = k + 1
                act_l(ia) = ll
                alive(ia) = .true.
            endif
            ! else: no match or lost conflict — track terminates (alive stays false)
        enddo

        ! Compact active list; recycle slots of terminated short tracks
        ib = 0
        do ia = 1, n_active
            if (alive(ia)) then
                ib = ib + 1
                act_track(ib) = act_track(ia)
                act_k(ib) = act_k(ia)
                act_l(ib) = act_l(ia)
            else
                ! Track terminated — check mindhr immediately
                it = act_track(ia)
                if (track_lengths(it) * dt_hours < mindhr) then
                    ! Too short: recycle slot
                    track_lengths(it) = 0
                    n_free = n_free + 1
                    free_slots(n_free) = it
                endif
            endif
        enddo
        n_active = ib

    enddo

    ! === Final: check remaining active tracks + compact ===
    ! Active tracks at end of data also need mindhr check
9000 continue
    do ia = 1, n_active
        it = act_track(ia)
        if (track_lengths(it) * dt_hours < mindhr) then
            track_lengths(it) = 0
        endif
    enddo

    ! Compact: remove gaps (recycled slots have track_lengths=0)
    ib = 0
    do it = 1, ntrack
        if (track_lengths(it) > 0) then
            ib = ib + 1
            if (ib /= it) then
                track_lon(ib, :) = track_lon(it, :)
                track_lat(ib, :) = track_lat(it, :)
                track_psl(ib, :) = track_psl(it, :)
                track_wspd(ib, :) = track_wspd(it, :)
                track_time_idx(ib, :) = track_time_idx(it, :)
                track_lengths(ib) = track_lengths(it)
            endif
        endif
    enddo
    ! Clear remaining slots
    do it = ib + 1, ntrack
        track_lon(it, :) = -999.
        track_lat(it, :) = -999.
        track_psl(it, :) = -999.
        track_wspd(it, :) = -999.
        track_time_idx(it, :) = 0
        track_lengths(it) = 0
    enddo
    ntrack = ib

end subroutine connect_tracks


subroutine calc_env_vorticity(u, v, vor, lonv, latv, nx, ny, missing)
    ! Calculate vorticity on a cropped subgrid region.
    !
    !f2py intent(in)  :: u, v, lonv, latv, nx, ny, missing
    !f2py intent(out) :: vor
    implicit none
    integer, intent(in) :: nx, ny
    real, intent(in)  :: u(nx, ny), v(nx, ny)
    real, intent(in)  :: lonv(nx), latv(ny)
    real, intent(in)  :: missing
    real, intent(out) :: vor(nx, ny)

    real, parameter :: pi = 3.141592
    real, parameter :: R  = 6371000.
    real :: dx, dy
    integer :: i, j

    vor = missing

    do j = 2, ny - 1
        ! Skip if adjacent latitudes are NaN/invalid (pole clamping)
        if (latv(j+1) == latv(j+1) .and. latv(j-1) == latv(j-1)) then
            dy = (latv(j+1) - latv(j-1)) * 2. * R * pi / 360.
        else
            cycle
        endif
        do i = 2, nx - 1
            if (v(i+1,j) /= missing .and. v(i-1,j) /= missing .and. &
                u(i,j+1) /= missing .and. u(i,j-1) /= missing) then
                dx = (lonv(i+1) - lonv(i-1)) * 2. * R * pi / 360. * cos(latv(j) * pi / 180.)
                if (abs(dx) > 1.) then
                    vor(i,j) = (v(i+1,j) - v(i-1,j)) / dx - (u(i,j+1) - u(i,j-1)) / dy
                endif
            endif
        enddo
    enddo

end subroutine calc_env_vorticity


subroutine azimuthal_mean_km(var, distkm, nx, ny, missing, bin_km, azmean, naz)
    ! Compute azimuthal mean using haversine distances in km bins.
    !
    !f2py intent(in)  :: var, distkm, nx, ny, missing, bin_km, naz
    !f2py intent(out) :: azmean
    implicit none
    integer, intent(in) :: nx, ny, naz
    real, intent(in)  :: var(nx, ny)
    real, intent(in)  :: distkm(nx, ny)
    real, intent(in)  :: missing, bin_km
    real, intent(out) :: azmean(0:naz-1)

    real :: nn(0:naz-1)
    integer :: i, j, kk

    azmean = 0.
    nn = 0.

    do i = 1, nx
        do j = 1, ny
            if (var(i, j) /= missing) then
                kk = int(distkm(i, j) / bin_km)
                if (kk < naz) then
                    azmean(kk) = azmean(kk) + var(i, j)
                    nn(kk) = nn(kk) + 1.
                endif
            endif
        enddo
    enddo

    ! First pass: convert sums to averages
    do kk = 0, naz - 1
        if (nn(kk) > 0.) then
            azmean(kk) = azmean(kk) / nn(kk)
        endif
    enddo
    ! Second pass: fill empty bins by interpolating neighbors
    do kk = 1, naz - 2
        if (nn(kk) == 0. .and. nn(kk-1) > 0. .and. nn(kk+1) > 0.) then
            azmean(kk) = (azmean(kk-1) + azmean(kk+1)) / 2.
        endif
    enddo

end subroutine azimuthal_mean_km


subroutine haversine_km(lon1, lat1, lon2, lat2, dist)
    ! Haversine distance in km.
    !
    !f2py intent(in)  :: lon1, lat1, lon2, lat2
    !f2py intent(out) :: dist
    implicit none
    real, intent(in)  :: lon1, lat1, lon2, lat2
    real, intent(out) :: dist
    real, parameter :: pi = 3.141592
    real :: dx, dy, a

    dy = (lat2 - lat1) * pi / 180.
    dx = abs(lon2 - lon1)
    if (dx >= 180.) dx = 360. - dx
    dx = dx * pi / 180.
    a = sin(dy * 0.5)**2 + cos(lat1 * pi / 180.) * cos(lat2 * pi / 180.) * sin(dx * 0.5)**2
    dist = 6371. * 2. * atan2(sqrt(a), sqrt(1. - a))

end subroutine haversine_km


subroutine calc_env_all(u850, v850, u250, v250, t250, t500, psl, &
                        distkm, lonv, latv, nx, ny, missing, bin_km, naz, &
                        vor_inner_km, warmcore_km, &
                        wspd_nbin, psl_nbin, outer_start, outer_end, &
                        wspd_az_max, vws_az_lg, vor_cnt, psl_az_cnt, &
                        warmcore_250, warmcore_250_500)
    ! Combined env parameter calculation: vorticity + derived fields +
    ! azimuthal means in a single grid scan.  Replaces 6 separate f2py calls.
    !
    !f2py intent(in) :: u850,v850,u250,v250,t250,t500,psl
    !f2py intent(in) :: distkm,lonv,latv,nx,ny,missing,bin_km,naz
    !f2py intent(in) :: vor_inner_km,warmcore_km
    !f2py intent(in) :: wspd_nbin,psl_nbin,outer_start,outer_end
    !f2py intent(out) :: wspd_az_max,vws_az_lg,vor_cnt,psl_az_cnt
    !f2py intent(out) :: warmcore_250,warmcore_250_500
    implicit none
    integer, intent(in) :: nx, ny, naz
    real, intent(in)  :: u850(nx,ny), v850(nx,ny)
    real, intent(in)  :: u250(nx,ny), v250(nx,ny)
    real, intent(in)  :: t250(nx,ny), t500(nx,ny), psl(nx,ny)
    real, intent(in)  :: distkm(nx,ny), lonv(nx), latv(ny)
    real, intent(in)  :: missing, bin_km
    real, intent(in)  :: vor_inner_km, warmcore_km
    integer, intent(in) :: wspd_nbin, psl_nbin, outer_start, outer_end
    real, intent(out) :: wspd_az_max, vws_az_lg, vor_cnt, psl_az_cnt
    real, intent(out) :: warmcore_250, warmcore_250_500

    real, parameter :: pi = 3.141592
    real, parameter :: R  = 6371000.

    ! Azimuthal mean accumulators (5 variables)
    real :: az_wspd(0:naz-1), nn_wspd(0:naz-1)
    real :: az_vws(0:naz-1),  nn_vws(0:naz-1)
    real :: az_psl(0:naz-1),  nn_psl(0:naz-1)
    real :: az_t250(0:naz-1), nn_t250(0:naz-1)
    real :: az_t25050(0:naz-1), nn_t25050(0:naz-1)

    ! Vorticity temporaries
    real :: vor(nx, ny)
    real :: dx_m, dy_m

    ! Local vars
    integer :: i, j, kk
    real :: wspd_val, vws_val, t25050_val, d
    real :: vor_sum, vor_nn
    real :: t250_max_wc, t25050_max_wc
    logical :: has_wc, has_wc2

    ! --- Phase 1: compute vorticity field ---
    vor = missing
    do j = 2, ny - 1
        if (latv(j+1) == latv(j+1) .and. latv(j-1) == latv(j-1)) then
            dy_m = (latv(j+1) - latv(j-1)) * 2. * R * pi / 360.
        else
            cycle
        endif
        do i = 2, nx - 1
            if (v850(i+1,j) /= missing .and. v850(i-1,j) /= missing .and. &
                u850(i,j+1) /= missing .and. u850(i,j-1) /= missing) then
                dx_m = (lonv(i+1) - lonv(i-1)) * 2. * R * pi / 360. * cos(latv(j) * pi / 180.)
                if (abs(dx_m) > 1.) then
                    vor(i,j) = (v850(i+1,j) - v850(i-1,j)) / dx_m &
                             - (u850(i,j+1) - u850(i,j-1)) / dy_m
                endif
            endif
        enddo
    enddo

    ! --- Phase 2: single grid scan for all azimuthal means + vor_cnt + warmcore max ---
    az_wspd = 0.; nn_wspd = 0.
    az_vws  = 0.; nn_vws  = 0.
    az_psl  = 0.; nn_psl  = 0.
    az_t250 = 0.; nn_t250 = 0.
    az_t25050 = 0.; nn_t25050 = 0.
    vor_sum = 0.; vor_nn = 0.
    t250_max_wc = -1e20; has_wc = .false.
    t25050_max_wc = -1e20; has_wc2 = .false.

    do i = 1, nx
        do j = 1, ny
            d = distkm(i, j)
            kk = int(d / bin_km)

            ! wspd850 azimuthal mean
            if (u850(i,j) /= missing .and. v850(i,j) /= missing) then
                wspd_val = sqrt(u850(i,j)**2 + v850(i,j)**2)
                if (kk < naz) then
                    az_wspd(kk) = az_wspd(kk) + wspd_val
                    nn_wspd(kk) = nn_wspd(kk) + 1.
                endif

                ! vws
                if (u250(i,j) /= missing .and. v250(i,j) /= missing) then
                    vws_val = sqrt((u250(i,j)-u850(i,j))**2 + (v250(i,j)-v850(i,j))**2)
                    if (kk < naz) then
                        az_vws(kk) = az_vws(kk) + vws_val
                        nn_vws(kk) = nn_vws(kk) + 1.
                    endif
                endif
            endif

            ! psl azimuthal mean
            if (psl(i,j) /= missing .and. kk < naz) then
                az_psl(kk) = az_psl(kk) + psl(i,j)
                nn_psl(kk) = nn_psl(kk) + 1.
            endif

            ! t250 azimuthal mean + warmcore max
            if (t250(i,j) /= missing) then
                if (kk < naz) then
                    az_t250(kk) = az_t250(kk) + t250(i,j)
                    nn_t250(kk) = nn_t250(kk) + 1.
                endif
                if (d <= warmcore_km) then
                    if (t250(i,j) > t250_max_wc) t250_max_wc = t250(i,j)
                    has_wc = .true.
                endif
            endif

            ! t250_500 mean + warmcore max
            if (t250(i,j) /= missing .and. t500(i,j) /= missing) then
                t25050_val = (t250(i,j) + t500(i,j)) / 2.
                if (kk < naz) then
                    az_t25050(kk) = az_t25050(kk) + t25050_val
                    nn_t25050(kk) = nn_t25050(kk) + 1.
                endif
                if (d <= warmcore_km) then
                    if (t25050_val > t25050_max_wc) t25050_max_wc = t25050_val
                    has_wc2 = .true.
                endif
            endif

            ! vorticity inner mean
            if (d < vor_inner_km .and. vor(i,j) /= missing) then
                vor_sum = vor_sum + vor(i,j)
                vor_nn = vor_nn + 1.
            endif
        enddo
    enddo

    ! --- Phase 3: finalize azimuthal means (average + interpolate empty bins) ---
    call finalize_azmean(az_wspd, nn_wspd, naz)
    call finalize_azmean(az_vws,  nn_vws,  naz)
    call finalize_azmean(az_psl,  nn_psl,  naz)
    call finalize_azmean(az_t250, nn_t250, naz)
    call finalize_azmean(az_t25050, nn_t25050, naz)

    ! --- Phase 4: extract final metrics ---
    ! wspd_az_max: max of azimuthal mean within wspd_nbin
    wspd_az_max = az_wspd(0)
    do kk = 1, wspd_nbin
        if (az_wspd(kk) > wspd_az_max) wspd_az_max = az_wspd(kk)
    enddo

    ! vws_az_lg: mean of azimuthal mean from outer_start to outer_end
    vws_az_lg = 0.
    do kk = outer_start, outer_end
        vws_az_lg = vws_az_lg + az_vws(kk)
    enddo
    vws_az_lg = vws_az_lg / real(outer_end - outer_start + 1)

    ! vor_cnt: inner vorticity mean × 1e5
    if (vor_nn > 0.) then
        vor_cnt = vor_sum / vor_nn * 1e5
    else
        vor_cnt = 0.
    endif

    ! psl_az_cnt: mean of azimuthal mean within psl_nbin, in hPa
    psl_az_cnt = 0.
    do kk = 0, psl_nbin
        psl_az_cnt = psl_az_cnt + az_psl(kk)
    enddo
    psl_az_cnt = psl_az_cnt / real(psl_nbin + 1) / 100.

    ! warmcore_250: max_wc - outer mean
    if (has_wc) then
        warmcore_250 = t250_max_wc
    else
        warmcore_250 = 0.
    endif
    ! Subtract outer azimuthal mean
    vws_az_lg = vws_az_lg  ! (already computed)
    ! Compute outer t250 mean
    warmcore_250 = warmcore_250
    do kk = outer_start, outer_end
        warmcore_250 = warmcore_250 - az_t250(kk) / real(outer_end - outer_start + 1)
    enddo
    if (.not. has_wc) warmcore_250 = 0.

    ! warmcore_250_500
    if (has_wc2) then
        warmcore_250_500 = t25050_max_wc
    else
        warmcore_250_500 = 0.
    endif
    do kk = outer_start, outer_end
        warmcore_250_500 = warmcore_250_500 - az_t25050(kk) / real(outer_end - outer_start + 1)
    enddo
    if (.not. has_wc2) warmcore_250_500 = 0.

end subroutine calc_env_all


subroutine finalize_azmean(azmean, nn, naz)
    ! Average + interpolate empty bins. Internal helper.
    implicit none
    integer, intent(in) :: naz
    real, intent(inout) :: azmean(0:naz-1)
    real, intent(in)    :: nn(0:naz-1)
    integer :: kk

    do kk = 0, naz - 1
        if (nn(kk) > 0.) then
            azmean(kk) = azmean(kk) / nn(kk)
        endif
    enddo
    do kk = 1, naz - 2
        if (nn(kk) == 0. .and. nn(kk-1) > 0. .and. nn(kk+1) > 0.) then
            azmean(kk) = (azmean(kk-1) + azmean(kk+1)) / 2.
        endif
    enddo
end subroutine finalize_azmean
