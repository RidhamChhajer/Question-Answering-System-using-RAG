import styles from './PipelineStatus.module.css';

export default function PipelineStatus({ steps }) {
  if (!steps.length) return null;
  return (
    <div className={styles.wrap}>
      {steps.map((step, i) => (
        <div key={i} className={`${styles.step} ${styles[step.status]}`}>
          <span className={styles.icon}>
            {step.status === 'done'   && '✓'}
            {step.status === 'active' && <span className={styles.spinner}/>}
            {step.status === 'pending'&& '·'}
          </span>
          <span className={styles.label}>{step.content}</span>
          {step.eta && step.status !== 'done' && (
            <span className={styles.eta}>{step.eta}</span>
          )}
        </div>
      ))}
    </div>
  );
}
