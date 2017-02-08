;; based on gosu-mode
;; https://github.com/gosu-lang/gosu-lang.org/tree/master/emacs
(require 'generic-x)

(defvar bi-indent-offset 2
  "Indentation offset for `bi-mode'.")

(defun generic-indent-line ()
  "Indent current line for any balanced-paren-mode'."
  (interactive)
  (let ((indent-col 0)
        (indentation-increasers "{")
        (indentation-decreasers "}")
        )
    (save-excursion
      (beginning-of-line)
      (condition-case nil
          (while t
            (backward-up-list 1)
            (when (looking-at indentation-increasers)
              (setq indent-col (+ indent-col bi-indent-offset))))
        (error nil)))
    (save-excursion
      (back-to-indentation)
      (when (and (looking-at indentation-decreasers) (>= indent-col bi-indent-offset))
        (setq indent-col (- indent-col bi-indent-offset))))
    (indent-line-to indent-col)))

;; Set up the actual generic mode
(define-generic-mode 'bi-mode
  ;; comment-list
  '("//")
  ;; keyword-list
  '(
    "model"
    "dim"
    "const"
    "param"
    "input"
    "state"
    "obs"
    "noise"
    "sub"
    "transition"
    "ode"
    "parameter"
    "proposal_parameter"
    "initial"
    "proposal_initial"
    "observation"
    "abs"
    "acos"
    "acosh"
    "asin"
    "asinh"
    "atan"
    "atan2"
    "atanh"
    "ceil"
    "cos"
    "cosh"
    "erf"
    "erfc"
    "exp"
    "floor"
    "gamma"
    "lgamma"
    "log"
    "max"
    "min"
    "mod"
    "pow"
    "round"
    "sin"
    "sinh"
    "sqrt"
    "tan"
    "tanh"
    )
  ;; font-lock-list
  '(("=" . 'font-lock-type-face)
    ("<-" . 'font-lock-type-face)
    ("\\[[^]]+\\]" . 'font-lock-string-face))
;;    ("[-+\\*/]" . 'font-lock-negation-char-face)
;;    ("[^a-zA-Z]\\([0-9][0-9.]*\\)" 1 'font-lock-function-name-face))
;;    ("[a-zA-Z][a-zA-Z_0-9]*" . 'font-lock-function-name-face))
  ;; auto-mode-list
  '(".bi\\'")
  ;; function-list
  '( ;; env setup
    (lambda ()
      (make-local-variable 'generic-indent-line)
      (set 'indent-line-function 'generic-indent-line)
      ))
  "Bi mode"
  )

(provide 'bi-mode)
