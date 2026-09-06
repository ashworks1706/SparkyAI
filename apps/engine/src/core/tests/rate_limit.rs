//! Per-user request limiting.

#[test]
fn the_rate_limiter_counts_per_user_and_is_off_at_zero() {
    use crate::routes::rate_limit::RateLimiter;

    let limiter = RateLimiter::new(2);
    assert!(limiter.allow("a"));
    assert!(limiter.allow("a"));
    assert!(
        !limiter.allow("a"),
        "the third request in the window is refused"
    );
    assert!(limiter.allow("b"), "another caller has their own count");

    let off = RateLimiter::new(0);
    for _ in 0..100 {
        assert!(off.allow("a"));
    }
}
